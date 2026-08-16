"""StageRunner: Generic async wrapper for pipeline stages.

Each stage in the pipeline (VAD, ASR, MT, UI) follows the same pattern:
  1. Read a message from the input queue.
  2. Process it (call the engine).
  3. Measure latency.
  4. Put the result on the output queue.
  5. Handle errors without crashing the pipeline.

StageRunner encapsulates this pattern so individual stages don't need
to reimplement queue/error/metrics logic.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, TypeVar

import structlog

from translator.core.events import PipelineShutdown
from translator.pipeline.metrics import PipelineMetrics

logger = structlog.get_logger(__name__)

T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")


class StageRunner:
    """Generic async runner for a single pipeline stage.

    Wraps a processing function with queue consumption, error handling,
    metrics recording, and graceful shutdown via PipelineShutdown sentinel.
    """

    def __init__(
        self,
        name: str,
        input_queue: asyncio.Queue[Any],
        output_queue: asyncio.Queue[Any] | None,
        process_fn: Callable[..., Awaitable[Any]],
        metrics: PipelineMetrics,
        log_latencies: bool = True,
    ) -> None:
        """Initialize the stage runner.

        Args:
            name: Stage name for logging and metrics.
            input_queue: Queue to read messages from.
            output_queue: Queue to write results to (None for terminal stages).
            process_fn: Async callable that processes a single message.
            metrics: Shared metrics collector.
            log_latencies: Whether to log per-message latency.
        """
        self._name = name
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._process_fn = process_fn
        self._metrics = metrics
        self._log_latencies = log_latencies
        self._running = False
        self._processed_count = 0

    async def run(self) -> None:
        """Main loop: consume from input queue, process, produce to output queue.

        Runs until a PipelineShutdown sentinel is received, then forwards
        the sentinel downstream and exits.
        """
        self._running = True
        logger.info("stage_started", stage=self._name)

        try:
            while self._running:
                msg = await self._input_queue.get()

                # Check for shutdown sentinel
                if isinstance(msg, PipelineShutdown):
                    logger.info(
                        "stage_shutdown_received",
                        stage=self._name,
                        reason=msg.reason,
                    )
                    if self._output_queue is not None:
                        await self._output_queue.put(msg)
                    break

                # Process the message
                try:
                    start_ns = time.monotonic_ns()
                    result = await self._process_fn(msg)
                    elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000

                    # Record metrics
                    self._metrics.record_stage_latency(self._name, elapsed_ms)
                    self._processed_count += 1

                    # Measure queue wait time if the message has a timestamp
                    queue_wait_ms = 0.0
                    if hasattr(msg, "created_at_ns"):
                        queue_wait_ms = self._metrics.measure_queue_latency_ms(
                            msg.created_at_ns
                        )

                    if self._log_latencies:
                        logger.debug(
                            "stage_processed",
                            stage=self._name,
                            processing_ms=round(elapsed_ms, 2),
                            queue_wait_ms=round(queue_wait_ms, 2),
                            sequence_id=getattr(msg, "sequence_id", None),
                        )

                    # Forward result to output queue
                    if result is not None and self._output_queue is not None:
                        await self._output_queue.put(result)

                except Exception:
                    logger.exception(
                        "stage_processing_error",
                        stage=self._name,
                        message_type=type(msg).__name__,
                    )
                    # Continue processing — don't crash the pipeline

                finally:
                    self._input_queue.task_done()

        except asyncio.CancelledError:
            logger.info("stage_cancelled", stage=self._name)
        finally:
            self._running = False
            logger.info(
                "stage_stopped",
                stage=self._name,
                total_processed=self._processed_count,
            )

    def stop(self) -> None:
        """Signal the stage to stop after the current message."""
        self._running = False

    @property
    def processed_count(self) -> int:
        """Number of messages successfully processed."""
        return self._processed_count

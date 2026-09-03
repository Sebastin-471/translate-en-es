"""PipelineOrchestrator: Connects pipeline stages with typed async queues.

Creates the full data flow:
  AudioSource → [Queue] → VAD → [Queue] → ASR → [Queue] → MT → [Queue] → UI

Manages lifecycle: start all stages, run until stopped, graceful shutdown
with PipelineShutdown sentinel propagation.

Import rules:
  - This module imports ONLY from `translator.core` (interfaces, events, config).
  - It does NOT import any infrastructure implementations.
  - Concrete engines are injected via constructor (dependency injection).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from translator.core.events import (
    AudioChunk,
    PipelineShutdown,
    TranscriptResult,
    TranslationResult,
    VADSegment,
)
from translator.pipeline.metrics import PipelineMetrics
from translator.pipeline.stage_runner import StageRunner

if TYPE_CHECKING:
    from translator.core.config import AppConfig
    from translator.core.interfaces import (
        ASREngine,
        AudioSource,
        MTEngine,
        UIRenderer,
        VADEngine,
    )

logger = structlog.get_logger(__name__)


class PipelineOrchestrator:
    """Orchestrates the full translation pipeline.

    Receives injected engine implementations (via constructor), creates
    typed async queues between them, and manages the pipeline lifecycle.

    This class imports ONLY from core/ — it has zero knowledge of which
    concrete implementations are being used. The composition root (app.py)
    is responsible for creating concrete engines and passing them here.
    """

    def __init__(
        self,
        audio_source: AudioSource,
        vad_engine: VADEngine,
        asr_engine: ASREngine,
        mt_engine: MTEngine,
        ui_renderer: UIRenderer,
        config: AppConfig,
    ) -> None:
        self._audio_source = audio_source
        self._vad_engine = vad_engine
        self._asr_engine = asr_engine
        self._mt_engine = mt_engine
        self._ui_renderer = ui_renderer
        self._config = config

        queue_size = config.pipeline.queue_size

        # Typed async queues between stages
        self._audio_queue: asyncio.Queue[AudioChunk | PipelineShutdown] = asyncio.Queue(
            maxsize=queue_size
        )
        self._vad_queue: asyncio.Queue[VADSegment | PipelineShutdown] = asyncio.Queue(
            maxsize=queue_size
        )
        self._transcript_queue: asyncio.Queue[
            TranscriptResult | PipelineShutdown
        ] = asyncio.Queue(maxsize=queue_size)
        self._translation_queue: asyncio.Queue[
            TranslationResult | PipelineShutdown
        ] = asyncio.Queue(maxsize=queue_size)

        # Metrics
        self._metrics = PipelineMetrics()

        # Task references for lifecycle management
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False
        self._paused = asyncio.Event()
        self._paused.set()  # Start in un-paused state (event is "set" = running)

    async def start(self) -> None:
        """Start all pipeline stages and the audio capture loop."""
        logger.info("pipeline_starting")

        # Start UI renderer
        await self._ui_renderer.start()

        # Start audio source
        await self._audio_source.start()

        # Create stage runners for VAD → ASR → MT → UI
        vad_runner = StageRunner(
            name="vad",
            input_queue=self._audio_queue,
            output_queue=self._vad_queue,
            process_fn=self._process_vad,
            metrics=self._metrics,
            log_latencies=self._config.logging.log_latencies,
        )

        asr_runner = StageRunner(
            name="asr",
            input_queue=self._vad_queue,
            output_queue=self._transcript_queue,
            process_fn=self._asr_engine.transcribe,
            metrics=self._metrics,
            log_latencies=self._config.logging.log_latencies,
        )

        mt_runner = StageRunner(
            name="mt",
            input_queue=self._transcript_queue,
            output_queue=self._translation_queue,
            process_fn=self._mt_engine.translate,
            metrics=self._metrics,
            log_latencies=self._config.logging.log_latencies,
        )

        ui_runner = StageRunner(
            name="ui",
            input_queue=self._translation_queue,
            output_queue=None,  # Terminal stage
            process_fn=self._ui_renderer.show,
            metrics=self._metrics,
            log_latencies=self._config.logging.log_latencies,
        )

        # Launch all stages as async tasks
        self._running = True
        self._tasks = [
            asyncio.create_task(self._audio_capture_loop(), name="audio_capture"),
            asyncio.create_task(vad_runner.run(), name="vad_stage"),
            asyncio.create_task(asr_runner.run(), name="asr_stage"),
            asyncio.create_task(mt_runner.run(), name="mt_stage"),
            asyncio.create_task(ui_runner.run(), name="ui_stage"),
        ]

        logger.info("pipeline_started", stages=len(self._tasks))

    async def run(self) -> None:
        """Run the pipeline until stopped or an error occurs.

        Blocks until all tasks complete (either via stop() or error).
        """
        if not self._tasks:
            raise RuntimeError("Pipeline not started — call start() first")

        try:
            # Wait for all tasks — if any raises, others continue
            await asyncio.gather(*self._tasks, return_exceptions=True)
        except asyncio.CancelledError:
            logger.info("pipeline_cancelled")
        finally:
            self._metrics.log_summary()

    async def stop(self) -> None:
        """Gracefully shut down the pipeline.

        Sends a PipelineShutdown sentinel through the pipeline so each
        stage can finish its current work and propagate the shutdown
        downstream.
        """
        if not self._running:
            return

        logger.info("pipeline_stopping")
        self._running = False

        # Stop audio capture (it will put the shutdown sentinel)
        await self._audio_source.stop()

        # Send shutdown sentinel into the first queue
        await self._audio_queue.put(PipelineShutdown(reason="user_stop"))

        # Wait for all tasks to finish (with timeout)
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=10.0,
            )
        except TimeoutError:
            logger.warning("pipeline_shutdown_timeout", timeout_s=10)
            for task in self._tasks:
                if not task.done():
                    task.cancel()

        # Stop UI renderer
        await self._ui_renderer.stop()

        self._metrics.log_summary()
        logger.info("pipeline_stopped")

    @property
    def metrics(self) -> PipelineMetrics:
        """Access the pipeline metrics collector."""
        return self._metrics

    @property
    def is_paused(self) -> bool:
        """Return True if the pipeline is currently paused."""
        return not self._paused.is_set()

    async def pause(self) -> None:
        """Pause the audio capture loop. Stages stay alive but starve."""
        if not self._paused.is_set():
            return  # Already paused
        self._paused.clear()
        logger.info("pipeline_paused")

    async def resume(self) -> None:
        """Resume the audio capture loop."""
        if self._paused.is_set():
            return  # Already running
        self._paused.set()
        logger.info("pipeline_resumed")

    async def toggle_pause(self) -> None:
        """Toggle between paused and running states."""
        if self.is_paused:
            await self.resume()
        else:
            await self.pause()

    # --- Internal processing methods ---

    async def _audio_capture_loop(self) -> None:
        """Continuously capture audio and feed it into the pipeline."""
        logger.info("audio_capture_started")
        try:
            while self._running:
                # Wait if paused
                await self._paused.wait()

                try:
                    chunk = await self._audio_source.read_chunk()
                    await self._audio_queue.put(chunk)
                except StopAsyncIteration:
                    logger.info("audio_source_exhausted")
                    await self._audio_queue.put(PipelineShutdown(reason="audio_exhausted"))
                    break
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("audio_capture_error")
                    await asyncio.sleep(0.1)  # Back off on error

        except asyncio.CancelledError:
            logger.info("audio_capture_cancelled")

    async def _process_vad(self, chunk: AudioChunk) -> VADSegment | None:
        """Process a chunk through the VAD engine.

        Returns a VADSegment if speech was detected, None otherwise.
        StageRunner will only forward non-None results to the output queue.
        """
        return await self._vad_engine.process_chunk(chunk)

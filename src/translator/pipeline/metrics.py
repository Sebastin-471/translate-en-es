"""Pipeline latency metrics collection.

Measures and tracks per-stage latency across the pipeline using
monotonic timestamps embedded in each message dataclass. Provides
percentile statistics for diagnostics.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class StageMetrics:
    """Latency metrics for a single pipeline stage.

    Attributes:
        stage_name: Human-readable name of the stage.
        latencies_ms: List of observed latency values (ms).
    """

    stage_name: str
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, latency_ms: float) -> None:
        """Record a latency observation."""
        self.latencies_ms.append(latency_ms)

    @property
    def count(self) -> int:
        return len(self.latencies_ms)

    @property
    def p50(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)

    @property
    def p95(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return self._percentile(0.95)

    @property
    def p99(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return self._percentile(0.99)

    @property
    def mean(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)

    def _percentile(self, p: float) -> float:
        sorted_data = sorted(self.latencies_ms)
        idx = int(len(sorted_data) * p)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]

    def summary(self) -> dict[str, float]:
        """Return a summary dict of the metrics."""
        return {
            "count": float(self.count),
            "mean_ms": round(self.mean, 2),
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
        }


class PipelineMetrics:
    """Aggregates latency metrics across all pipeline stages.

    Each stage records its processing latency. The metrics collector
    also tracks end-to-end latency from audio capture to UI display.
    """

    def __init__(self) -> None:
        self._stages: dict[str, StageMetrics] = {}
        self._e2e = StageMetrics(stage_name="end_to_end")

    def get_stage(self, name: str) -> StageMetrics:
        """Get or create metrics for a stage."""
        if name not in self._stages:
            self._stages[name] = StageMetrics(stage_name=name)
        return self._stages[name]

    def record_stage_latency(self, stage_name: str, latency_ms: float) -> None:
        """Record a latency observation for a stage."""
        metrics = self.get_stage(stage_name)
        metrics.record(latency_ms)

    def record_e2e_latency(self, latency_ms: float) -> None:
        """Record an end-to-end pipeline latency observation."""
        self._e2e.record(latency_ms)

    def log_summary(self) -> None:
        """Log a summary of all stage metrics."""
        for name, metrics in self._stages.items():
            if metrics.count > 0:
                logger.info(
                    "pipeline_metrics",
                    stage=name,
                    **metrics.summary(),
                )

        if self._e2e.count > 0:
            logger.info(
                "pipeline_metrics",
                stage="end_to_end",
                **self._e2e.summary(),
            )

    def get_all_summaries(self) -> dict[str, dict[str, float]]:
        """Return all stage summaries as a dict."""
        result: dict[str, dict[str, float]] = {}
        for name, metrics in self._stages.items():
            result[name] = metrics.summary()
        result["end_to_end"] = self._e2e.summary()
        return result

    @staticmethod
    def measure_queue_latency_ms(created_at_ns: int) -> float:
        """Measure how long a message waited in a queue.

        Args:
            created_at_ns: The monotonic timestamp (ns) from the message.

        Returns:
            Queue wait time in milliseconds.
        """
        return (time.monotonic_ns() - created_at_ns) / 1_000_000

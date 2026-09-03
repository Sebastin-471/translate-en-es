"""Prometheus metrics for translate-en-es."""

from __future__ import annotations

import structlog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge, Histogram

logger = structlog.get_logger(__name__)

# Global metrics registry (initialized lazily)
_pipeline_latency: "Histogram | None" = None
_pipeline_stage_latency: "Histogram | None" = None
_audio_chunks_processed: "Counter | None" = None
_vad_segments_emitted: "Counter | None" = None
_transcripts_generated: "Counter | None" = None
_translations_generated: "Counter | None" = None
_ui_updates: "Counter | None" = None
_errors_total: "Counter | None" = None
_active_pipeline: "Gauge | None" = None
_vram_usage_mb: "Gauge | None" = None
_models_loaded: "Gauge | None" = None


def init_metrics(enabled: bool = True) -> None:
    """Initialize Prometheus metrics.

    Args:
        enabled: Whether to enable metrics collection.
    """
    global (
        _pipeline_latency,
        _pipeline_stage_latency,
        _audio_chunks_processed,
        _vad_segments_emitted,
        _transcripts_generated,
        _translations_generated,
        _ui_updates,
        _errors_total,
        _active_pipeline,
        _vram_usage_mb,
        _models_loaded,
    )

    if not enabled:
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram
    except ImportError:
        logger.warning("prometheus_client_not_installed_metrics_disabled")
        return

    _pipeline_latency = Histogram(
        "translate_pipeline_latency_seconds",
        "End-to-end pipeline latency in seconds",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    _pipeline_stage_latency = Histogram(
        "translate_pipeline_stage_latency_seconds",
        "Per-stage pipeline latency in seconds",
        ["stage"],  # vad, asr, mt, ui
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )

    _audio_chunks_processed = Counter(
        "translate_audio_chunks_processed_total",
        "Total number of audio chunks processed",
    )

    _vad_segments_emitted = Counter(
        "translate_vad_segments_emitted_total",
        "Total number of VAD segments emitted",
        ["type"],  # partial, final
    )

    _transcripts_generated = Counter(
        "translate_transcripts_generated_total",
        "Total number of transcripts generated",
        ["type"],  # partial, final
    )

    _translations_generated = Counter(
        "translate_translations_generated_total",
        "Total number of translations generated",
        ["type"],  # partial, final
    )

    _ui_updates = Counter(
        "translate_ui_updates_total",
        "Total number of UI updates",
    )

    _errors_total = Counter(
        "translate_errors_total",
        "Total number of errors",
        ["component", "error_type"],
    )

    _active_pipeline = Gauge(
        "translate_pipeline_active",
        "Whether the pipeline is currently running (1) or stopped (0)",
    )

    _vram_usage_mb = Gauge(
        "translate_vram_usage_mb",
        "Current VRAM usage in MB",
    )

    _models_loaded = Gauge(
        "translate_models_loaded",
        "Number of models currently loaded",
    )

    logger.info("prometheus_metrics_initialized")


def observe_pipeline_latency(seconds: float) -> None:
    """Observe end-to-end pipeline latency."""
    if _pipeline_latency:
        _pipeline_latency.observe(seconds)


def observe_stage_latency(stage: str, seconds: float) -> None:
    """Observe per-stage pipeline latency."""
    if _pipeline_stage_latency:
        _pipeline_stage_latency.labels(stage=stage).observe(seconds)


def increment_audio_chunks() -> None:
    """Increment audio chunks processed counter."""
    if _audio_chunks_processed:
        _audio_chunks_processed.inc()


def increment_vad_segments(segment_type: str) -> None:
    """Increment VAD segments counter."""
    if _vad_segments_emitted:
        _vad_segments_emitted.labels(type=segment_type).inc()


def increment_transcripts(transcript_type: str) -> None:
    """Increment transcripts counter."""
    if _transcripts_generated:
        _transcripts_generated.labels(type=transcript_type).inc()


def increment_translations(translation_type: str) -> None:
    """Increment translations counter."""
    if _translations_generated:
        _translations_generated.labels(type=translation_type).inc()


def increment_ui_updates() -> None:
    """Increment UI updates counter."""
    if _ui_updates:
        _ui_updates.inc()


def increment_errors(component: str, error_type: str) -> None:
    """Increment errors counter."""
    if _errors_total:
        _errors_total.labels(component=component, error_type=error_type).inc()


def set_pipeline_active(active: bool) -> None:
    """Set pipeline active gauge."""
    if _active_pipeline:
        _active_pipeline.set(1 if active else 0)


def set_vram_usage(mb: float) -> None:
    """Set VRAM usage gauge."""
    if _vram_usage_mb:
        _vram_usage_mb.set(mb)


def set_models_loaded(count: int) -> None:
    """Set models loaded gauge."""
    if _models_loaded:
        _models_loaded.set(count)


def get_metrics_registry() -> Any:
    """Get the Prometheus registry for exposing metrics."""
    try:
        from prometheus_client import REGISTRY
        return REGISTRY
    except ImportError:
        return None
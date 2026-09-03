"""Observability package for tracing and metrics."""

from __future__ import annotations

from translator.observability.metrics import (
    init_metrics,
    observe_pipeline_latency,
    observe_stage_latency,
    increment_audio_chunks,
    increment_vad_segments,
    increment_transcripts,
    increment_translations,
    increment_ui_updates,
    increment_errors,
    set_pipeline_active,
    set_vram_usage,
    set_models_loaded,
    get_metrics_registry,
)
from translator.observability.tracing import (
    init_tracing,
    get_tracer,
    is_tracing_enabled,
    trace_span,
    add_span_attributes,
    record_exception,
)

__all__ = [
    # Metrics
    "init_metrics",
    "observe_pipeline_latency",
    "observe_stage_latency",
    "increment_audio_chunks",
    "increment_vad_segments",
    "increment_transcripts",
    "increment_translations",
    "increment_ui_updates",
    "increment_errors",
    "set_pipeline_active",
    "set_vram_usage",
    "set_models_loaded",
    "get_metrics_registry",
    # Tracing
    "init_tracing",
    "get_tracer",
    "is_tracing_enabled",
    "trace_span",
    "add_span_attributes",
    "record_exception",
]
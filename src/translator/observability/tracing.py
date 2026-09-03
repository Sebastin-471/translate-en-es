"""OpenTelemetry tracing setup for distributed tracing."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

logger = structlog.get_logger(__name__)

_tracer: "Tracer | None" = None
_initialized = False


def init_tracing(
    service_name: str = "translate-en-es",
    otlp_endpoint: str | None = None,
    enabled: bool = True,
) -> None:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Name of the service for tracing.
        otlp_endpoint: OTLP gRPC endpoint (e.g., "http://localhost:4317").
                       If None, reads from OTEL_EXPORTER_OTLP_ENDPOINT env var.
        enabled: Whether to enable tracing.
    """
    global _tracer, _initialized

    if _initialized or not enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("opentelemetry_not_installed_tracing_disabled")
        return

    # Create resource with service name
    resource = Resource.create({"service.name": service_name})

    # Create tracer provider
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # Configure OTLP exporter
    endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        logger.info("opentelemetry_configured", endpoint=endpoint)
    else:
        logger.info("opentelemetry_configured_no_exporter", note="Set OTEL_EXPORTER_OTLP_ENDPOINT to export traces")

    _tracer = trace.get_tracer(service_name)
    _initialized = True
    logger.info("tracing_initialized", service_name=service_name)


def get_tracer() -> "Tracer | None":
    """Get the configured tracer."""
    return _tracer


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled."""
    return _initialized and _tracer is not None


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None):
    """Context manager for creating a traced span.

    Usage:
        with trace_span("my_operation", {"key": "value"}) as span:
            do_work()
    """
    tracer = get_tracer()
    if not tracer:
        yield None
        return

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span


def add_span_attributes(attributes: dict[str, Any]) -> None:
    """Add attributes to the current span."""
    tracer = get_tracer()
    if not tracer:
        return

    from opentelemetry import trace
    span = trace.get_current_span()
    if span and span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)


def record_exception(exc: Exception) -> None:
    """Record an exception in the current span."""
    tracer = get_tracer()
    if not tracer:
        return

    from opentelemetry import trace
    span = trace.get_current_span()
    if span and span.is_recording():
        span.record_exception(exc)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
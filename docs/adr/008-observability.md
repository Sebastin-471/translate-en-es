# ADR-008: OpenTelemetry Tracing and Prometheus Metrics

## Status
Accepted

## Context
Production observability requires:
- Distributed tracing across pipeline stages (latency, errors)
- Metrics for alerting (queue depths, VRAM, error rates)
- Standard formats (OTLP, Prometheus) for integration with Grafana, Jaeger, etc.

## Decision
Optional observability stack (`src/translator/observability/`):

**Tracing** (`tracing.py`):
- OpenTelemetry SDK with OTLP gRPC exporter
- `trace_span()` context manager for pipeline stages
- Automatic `sequence_id` propagation via span attributes
- Disabled by default; enabled via `OTEL_EXPORTER_OTLP_ENDPOINT`

**Metrics** (`metrics.py`):
- Prometheus client with standard metrics:
  - `translate_pipeline_latency_seconds` (histogram)
  - `translate_pipeline_stage_latency_seconds` (histogram, by stage)
  - `translate_audio_chunks_processed_total` (counter)
  - `translate_vad_segments_emitted_total` (counter, by type)
  - `translate_errors_total` (counter, by component)
  - `translate_vram_usage_mb` (gauge)
- `/metrics` endpoint via `prometheus_client` (future: HTTP server)

Integration in `logging/setup.py` - initialized once at startup.

## Consequences

### Positive
- **Vendor Neutral**: OTLP/Prometheus work with any backend
- **Low Overhead**: Disabled by default; minimal when enabled
- **Actionable**: Per-stage latency, error categorization
- **Standards Compliant**: Works with Grafana, Jaeger, Datadog, etc.

### Negative
- **Optional Dependency**: Extra install (`pip install -e ".[observability]"`)
- **Cardinality**: Stage labels low cardinality; avoid high-cardinality labels

### Risks
- OTLP exporter may block on network issues (use async exporter in future)
- Metric cardinality explosion if not careful with labels
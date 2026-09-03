# translate-en-es

Real-time system audio translation (English to Spanish) for video calls (Teams, Zoom, Meet, etc.).

This system uses a modular pipeline:
1. Audio capture (WASAPI Loopback on Windows / PipeWire on Linux)
2. VAD (Voice Activity Detection using Silero VAD)
3. ASR (Speech-to-Text using faster-whisper)
4. MT (Machine Translation EN→ES using MarianMT via CTranslate2)
5. UI (Always-on-top subtitle overlay)

## Quick Start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install -e ".[dev]"
python -m translator.app --mock  # No GPU/downloads needed
```

## Documentation

- **Architecture**: [docs/adr/](docs/adr/) - Architecture Decision Records
- **Configuration**: [config/base.yaml](config/base.yaml) + environment overlays
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **API Reference**: Run `make -C docs html` to generate

## Running

```bash
# Production (real models, downloads on first run)
python -m translator.app --config config/base.yaml --env production

# Development (mock engines, hot-reload, DEBUG logs)
python -m translator.app --config config/base.yaml --env development

# Mock mode only (no models, fast iteration)
python -m translator.app --mock

# Custom config
python -m translator.app --config my-config.yaml
```

> **Note:** First production run downloads ~1.5GB of models (Silero VAD, Faster-Whisper, MarianMT).

## Testing

```bash
# Unit tests (fast, no GPU)
pytest tests/unit -v

# Integration tests (mock pipeline)
pytest tests/integration -v

# All tests with coverage
pytest tests/ -v --cov=translator --cov-fail-under=80

# GPU tests (requires CUDA)
pytest tests/ -m gpu -v
```

## Model Preparation (Offline Deployment)

```bash
# Prepare all models with int8 quantization
python scripts/prepare_models.py --output-dir ./models

# Verify prepared models
python verify_models.py
```

## Observability

```bash
# Enable OpenTelemetry tracing
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
python -m translator.app

# Prometheus metrics (when HTTP server added)
curl http://localhost:9090/metrics
```

## Architecture Overview

The project follows **Hexagonal Architecture** (Ports & Adapters):

```
src/translator/
├── core/           # Pure domain: interfaces, events, config, plugins
├── pipeline/       # Orchestration: queues, stage runners, metrics
├── infrastructure/ # Adapters: WASAPI, Silero, Whisper, MarianMT, GPU
├── ui/             # Tkinter overlay, system tray, hotkeys
├── observability/  # OpenTelemetry tracing, Prometheus metrics
└── app.py          # Composition root (only imports infrastructure)
```

See [docs/adr/001-hexagonal-architecture.md](docs/adr/001-hexagonal-architecture.md) for details.

## Configuration

Environment-specific configs (deep-merged):
- `config/base.yaml` - Shared defaults
- `config/development.yaml` - Dev overrides (mock, DEBUG)
- `config/production.yaml` - Prod overrides (JSON logs, real engines)

Environment variables override all: `TRANSLATOR_ASR__MODEL_SIZE=medium`

## License

MIT License - see [LICENSE](LICENSE) for details.
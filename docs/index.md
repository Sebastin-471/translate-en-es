# Documentation

## Architecture Decision Records (ADRs)
See [docs/adr/](adr/README.md) for architectural decisions.

## API Reference
Generated via Sphinx (run `make -C docs html`)

## Configuration
See [config/base.yaml](../config/base.yaml) and environment overlays.

## Contributing
See [CONTRIBUTING.md](../CONTRIBUTING.md).

## Development Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pre-commit install
```

## Running Tests
```bash
pytest tests/unit        # Unit tests (fast, no GPU)
pytest tests/integration # Integration tests (mock mode)
pytest -m gpu            # GPU tests (requires CUDA)
```

## Model Preparation
```bash
python scripts/prepare_models.py --output-dir ./models
python verify_models.py
```

## Observability
```bash
# Enable tracing
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
python -m translator.app

# Prometheus metrics (when HTTP server added)
curl http://localhost:9090/metrics
```
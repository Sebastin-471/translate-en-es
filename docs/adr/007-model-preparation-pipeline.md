# ADR-007: Offline Model Preparation Pipeline

## Status
Accepted

## Context
First-run model downloads (1.5GB+) are slow and fail in air-gapped environments. Models need quantization (int8) for CPU/GPU efficiency. Production deployments require verified, reproducible model artifacts.

## Decision
`scripts/prepare_models.py` creates offline model packages:

1. **Download**: Pinned HF revisions via `huggingface_hub.snapshot_download`
2. **Convert**: `ct2-transformers-converter` → CTranslate2 format
3. **Quantize**: int8/float16 via converter flags
4. **Verify**: Load model, run inference, compute SHA256
5. **Manifest**: `model_manifest.json` with hashes, sizes, metadata
6. **Package**: Ready for bundling or distribution

`verify_models.py` validates against manifest at runtime.

## Consequences

### Positive
- **Air-Gapped Deploy**: Pre-baked models work offline
- **Reproducibility**: Pinned revisions + hashes = exact same models
- **Performance**: Pre-quantized models load faster
- **CI/CD Friendly**: Manifest verification in pipeline

### Negative
- **Storage**: Prepared models larger than HF cache (no compression)
- **Maintenance**: Script updates needed for new model versions
- **Conversion Time**: Initial preparation takes minutes

### Risks
- HF repo changes breaking conversion (pinned revisions mitigate)
- CTranslate2 version compatibility with converter
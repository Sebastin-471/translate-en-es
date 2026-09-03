# ADR-003: Centralized GPU Model Manager

## Status
Accepted

## Context
Multiple ML models (Whisper, MarianMT, Silero VAD) share limited VRAM on consumer GPUs (e.g., RTX 3060 12GB). Loading all models simultaneously causes OOM. Models need to be loaded/unloaded dynamically (e.g., switching ASR model size). No standard library exists for this in Python ML ecosystem.

## Decision
Implement `GPUModelManager` (`src/translator/core/model_manager.py` protocol + `infrastructure/gpu/gpu_model_manager.py`):

- **Registration**: Engines register models with estimated VRAM usage at startup
- **Budget Enforcement**: `max_vram_mb` config limits total VRAM; loading fails if exceeded
- **Lazy/Preload**: Configurable preload at startup vs. lazy load on first use
- **Device Detection**: Auto-detects CUDA vs CPU; falls back to CPU if VRAM insufficient
- **Hot-Reload**: `unload_model()` + `load_model()` enables model switching without restart
- **VRAM Tracking**: `get_vram_status()` returns current usage for monitoring

Engines implement optional `ModelLifecycle` protocol for integration.

## Consequences

### Positive
- **OOM Prevention**: Hard budget enforcement prevents crashes
- **Flexibility**: Supports different model sizes, quantization levels
- **Hot-Reload**: Change ASR model size without app restart
- **Observability**: VRAM metrics exposed via Prometheus

### Negative
- **Estimation Accuracy**: VRAM estimates are approximate (may over/under-estimate)
- **Fragmentation**: `torch.cuda.empty_cache()` doesn't always reclaim all memory
- **Single-Process**: Doesn't coordinate across multiple processes

### Risks
- CUDA context issues if models loaded in wrong order
- VRAM estimation may be inaccurate for newer model architectures
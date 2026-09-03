# ADR-001: Hexagonal Architecture (Ports & Adapters)

## Status
Accepted

## Context
The project needs to support multiple audio backends (WASAPI, PipeWire), ASR engines (Whisper, potentially WhisperX), MT engines (MarianMT, potentially NLLB), and UI frameworks (Tkinter, potentially customtkinter). Without a clear architecture, the codebase would become tightly coupled and difficult to test or extend.

## Decision
Adopt Hexagonal Architecture (Ports & Adapters):

- **Core Domain** (`src/translator/core/`): Pure Python, zero external dependencies
  - Interfaces/Protocols (`interfaces.py`, `model_manager.py`)
  - Events/Dataclasses (`events.py`)
  - Configuration (`config.py`)
- **Infrastructure Adapters** (`src/translator/infrastructure/`): Concrete implementations
  - Audio: `WASAPIAudioSource`, `PipeWireAudioSource`, `FileAudioSource`
  - VAD: `SileroVADEngine`
  - ASR: `WhisperASREngine`
  - MT: `MarianMTEngine`
  - GPU: `GPUModelManager`
- **Composition Root** (`app.py`): Only place that imports from both core and infrastructure
- **Pipeline Orchestration** (`src/translator/pipeline/`): Uses only core interfaces

## Consequences

### Positive
- **Testability**: All infrastructure can be swapped with mocks for unit/integration tests
- **Extensibility**: New backends/engines added without modifying core or pipeline
- **Separation of Concerns**: Business logic isolated from framework details
- **Dependency Direction**: Infrastructure depends on core, never vice versa

### Negative
- **Indirection**: More files to navigate; factories needed in composition root
- **Boilerplate**: Protocols and factories add some code overhead

### Risks
- Developers may accidentally import infrastructure in core (mitigated by mypy/ruff rules)
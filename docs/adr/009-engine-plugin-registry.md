# ADR-009: Plugin Registry for Engine Implementations

## Status
Accepted

## Context
Adding a new engine (e.g., WhisperX ASR, NLLB MT) required modifying `app.py` composition root. This violates Open/Closed Principle and makes third-party extensions impossible without forking.

## Decision
`EngineRegistry` (`src/translator/core/plugins.py`):

- **Registration**: Engines register via `EngineRegistry.register_*()` at import time
- **Discovery**: Composition root uses `EngineRegistry.get_*()` by name
- **Built-ins**: `register_builtin_engines()` called once in `app.py`
- **Extensibility**: Third-party packages can register engines without touching core

Example:
```python
# In third_party_whisperx/__init__.py
from translator.core.plugins import EngineRegistry
from translator.core.interfaces import ASREngine

class WhisperXASREngine:
    ...

EngineRegistry.register_asr_engine("whisperx", lambda cfg, mm: WhisperXASREngine(cfg, mm))
```

Config references engine by name: `asr.engine: "whisperx"`

## Consequences

### Positive
- **Open/Closed**: Add engines without modifying composition root
- **Third-Party Friendly**: External packages can extend
- **Testability**: Easy to swap implementations in tests
- **Documentation**: `list_*_engines()` shows available options

### Negative
- **Global State**: Registry is module-level singleton
- **Import Order**: Registration must happen before `get_*()` calls
- **Name Collisions**: Two packages registering same name (overwrite flag)

### Risks
- Circular imports if engines import registry at module level (use lazy imports)
- Registry pollution in tests (use `overwrite=True` or reset)
# ADR-004: Environment-Specific Configuration with Hot-Reload

## Status
Accepted

## Context
Different environments need different configs:
- Development: Mock engines, DEBUG logging, fast iteration
- Production: Real engines, JSON logging, stability
- Testing: Fast mocks, controlled randomness

Configuration changes (e.g., VAD threshold, UI font) should apply without restart.

## Decision
Three-layer configuration with hot-reload:

1. **Base Config** (`config/base.yaml`): Shared defaults
2. **Environment Overlay** (`config/development.yaml`, `config/production.yaml`): Deep-merged overrides
3. **Environment Variables** (`TRANSLATOR_<SECTION>__<KEY>`): Highest priority, runtime overrides

Hot-reload via `watchfiles`:
- `ConfigWatcher` monitors base + env config files
- On change: reloads config, emits `ConfigChangeEvent`
- Application subscribes and reconfigures affected components (logging, VAD, UI)

## Consequences

### Positive
- **No Restart Needed**: Tweak VAD threshold, UI colors live
- **Environment Parity**: Same codebase runs dev/prod with different configs
- **12-Factor Compatible**: Env vars for secrets/overrides
- **Type Safety**: Dataclasses with validation catch errors early

### Negative
- **Partial Reload**: Some changes (audio backend, model) still require restart
- **Complexity**: Deep merge, event propagation, thread safety

### Risks
- Config drift between environments if overlays not maintained
- Hot-reload loops if config writer triggers watcher (debounce mitigates)
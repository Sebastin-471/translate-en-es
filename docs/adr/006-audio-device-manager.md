# ADR-006: Audio Device Manager for Enumeration and Fallback

## Status
Accepted

## Context
Audio devices change dynamically: headphones plugged/unplugged, default device switched, Bluetooth connected. The app must:
- List available loopback devices for user selection
- Detect device changes and notify user
- Fallback to alternative backend/device on failure

## Decision
Introduce `AudioDeviceManager` protocol with platform implementations:

- **WASAPI** (`wasapi_device_manager.py`): Enumerates via PyAudioWPatch; polling-based change detection
- **PipeWire** (`pipewire_device_manager.py`): Uses `pactl subscribe` for real-time events; `pw-dump` fallback

`AudioDeviceInfo` dataclass provides unified device metadata (id, name, type, backend, default status).

Factory (`device_manager_factory.py`) returns platform-appropriate implementation.

## Consequences

### Positive
- **Cross-Platform**: Same API for Windows/Linux
- **User Choice**: Settings UI can show device list
- **Resilience**: Auto-fallback on device loss (future work)
- **Observability**: Device changes logged and traceable

### Negative
- **Polling on Windows**: `pactl subscribe` not available on WASAPI; polling every 2s
- **Permissions**: May need elevated privileges for some device queries

### Risks
- Device ID instability across reboots (use name matching as fallback)
- Platform-specific quirks in device enumeration
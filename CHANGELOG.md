# Changelog

## [0.1.0] - 2026-08-16

### ✨ New Features
- **Real-time Pipeline**: Core architecture for capturing system audio, running Voice Activity Detection (VAD), Automatic Speech Recognition (ASR), and Machine Translation (MT) in real-time.
- **Always-on-top UI**: Configurable Tkinter-based translucent overlay for displaying translated subtitles.
- **Platform Support**: Modular audio capture backend supporting Windows (WASAPI Loopback) and Linux (PipeWire).
- **GPU Resource Management**: Centralized VRAM budgeting to support graceful degradation on limited hardware (e.g., RTX 3060 12GB).
- **Mock Engines**: Complete suite of mock engines allowing development and testing without requiring heavy ML models or GPU access.
- **Global Hotkeys**: Cross-platform global hotkey management using `pynput` for toggling UI and pipeline state.

### 🔧 Improvements
- **Typed Asynchronous Architecture**: Fully decoupled pipeline using Python `typing.Protocol`, `dataclasses`, and `asyncio.Queue`.
- **Structured Logging**: Deep observability with ISO 8601 timestamps and latency metrics tracking (p50, p95, p99) per pipeline stage.
- **Config Management**: YAML-based configuration with environment variable override support.

### 🐛 Fixes
- Initial Release.

# translate-en-es

Real-time system audio translation (English to Spanish) for video calls (Teams, Zoom, Meet, etc.).
This system uses a modular pipeline:
1. Audio capture (WASAPI Loopback on Windows / PipeWire on Linux)
2. VAD (Voice Activity Detection using Silero VAD)
3. ASR (Speech-to-Text using faster-whisper)
4. MT (Machine Translation EN->ES using MarianMT via CTranslate2)
5. UI (Always-on-top subtitle overlay)

## Setup

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

## Running

```bash
python -m translator.app --config config.yaml
```

> **Note:** The first time you run the application without `--mock`, it will automatically download and cache the AI models (Silero VAD, Faster-Whisper, and MarianMT). This may take several minutes depending on your internet connection (approx. 1.5GB total).

To run in mock mode (simulated engines, no GPU or downloads required):
```bash
python -m translator.app --mock
```

## Testing

```bash
pytest tests/unit/
pytest tests/integration/
```

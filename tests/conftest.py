"""Shared test fixtures for unit and integration tests.

Provides pre-configured mock engines, test configs, sample audio data,
and pre-populated queues for testing pipeline stages in isolation.
"""

from __future__ import annotations

import asyncio
import struct
import math
from pathlib import Path

import pytest

from translator.core.config import (
    AppConfig,
    ASRConfig,
    AudioConfig,
    GPUConfig,
    HotkeyConfig,
    LoggingConfig,
    MTConfig,
    PipelineConfig,
    UIConfig,
    VADConfig,
)
from translator.core.events import (
    AudioChunk,
    TranscriptResult,
    TranslationResult,
    VADSegment,
)


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------


@pytest.fixture
def test_config() -> AppConfig:
    """Minimal test configuration with mock mode enabled."""
    return AppConfig(
        audio=AudioConfig(backend="file", sample_rate=16000, chunk_duration_ms=30),
        vad=VADConfig(threshold=0.5, min_speech_duration_ms=100, min_silence_duration_ms=100),
        asr=ASRConfig(model_size="tiny", device="cpu", language="en"),
        mt=MTConfig(
            model_name="Helsinki-NLP/opus-mt-en-es",
            source_language="en",
            target_language="es",
            device="cpu",
        ),
        gpu=GPUConfig(max_vram_mb=0),
        ui=UIConfig(enabled=False),
        hotkeys=HotkeyConfig(),
        logging=LoggingConfig(level="DEBUG", format="console"),
        pipeline=PipelineConfig(queue_size=10, mock_mode=True, mock_delay_ms=10),
    )


# ---------------------------------------------------------------------------
# Sample audio data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pcm_silence() -> bytes:
    """1 second of 16kHz 16-bit mono silence."""
    return b"\x00\x00" * 16000


@pytest.fixture
def sample_pcm_tone() -> bytes:
    """1 second of 16kHz 16-bit mono 440Hz sine wave."""
    n_samples = 16000
    samples: list[int] = []
    for i in range(n_samples):
        val = math.sin(2 * math.pi * 440 * i / 16000)
        samples.append(int(val * 16000))
    return struct.pack(f"<{n_samples}h", *samples)


@pytest.fixture
def sample_audio_chunk(sample_pcm_silence: bytes) -> AudioChunk:
    """A single AudioChunk of 30ms silence."""
    chunk_samples = int(16000 * 0.030)
    chunk_bytes = chunk_samples * 2
    return AudioChunk(
        data=sample_pcm_silence[:chunk_bytes],
        sample_rate=16000,
        channels=1,
        duration_ms=30.0,
    )


@pytest.fixture
def sample_vad_segment(sample_pcm_tone: bytes) -> VADSegment:
    """A VADSegment containing 1 second of speech."""
    return VADSegment(
        audio_data=sample_pcm_tone,
        sample_rate=16000,
        start_time_ms=0.0,
        end_time_ms=1000.0,
        duration_ms=1000.0,
        confidence=0.95,
    )


@pytest.fixture
def sample_transcript() -> TranscriptResult:
    """A sample transcript result."""
    return TranscriptResult(
        text="Hello, how are you today?",
        language="en",
        confidence=0.92,
        segment_start_ms=0.0,
        segment_end_ms=1000.0,
        processing_time_ms=150.0,
    )


@pytest.fixture
def sample_translation() -> TranslationResult:
    """A sample translation result."""
    return TranslationResult(
        original_text="Hello, how are you today?",
        translated_text="Hola, ¿cómo estás hoy?",
        source_language="en",
        target_language="es",
        segment_start_ms=0.0,
        segment_end_ms=1000.0,
        processing_time_ms=50.0,
    )


# ---------------------------------------------------------------------------
# Mock engines
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_audio_source(test_config: AppConfig) -> "MockAudioSource":
    """Pre-configured MockAudioSource."""
    from translator.infrastructure.mock.mock_audio import MockAudioSource

    return MockAudioSource(test_config.audio)


@pytest.fixture
def mock_vad_engine() -> "MockVADEngine":
    """Pre-configured MockVADEngine with short segment interval."""
    from translator.infrastructure.mock.mock_vad import MockVADEngine

    return MockVADEngine(chunks_per_segment=5, delay_ms=1.0)


@pytest.fixture
def mock_asr_engine() -> "MockASREngine":
    """Pre-configured MockASREngine."""
    from translator.infrastructure.mock.mock_asr import MockASREngine

    return MockASREngine(delay_ms=10.0, language="en")


@pytest.fixture
def mock_mt_engine() -> "MockMTEngine":
    """Pre-configured MockMTEngine."""
    from translator.infrastructure.mock.mock_mt import MockMTEngine

    return MockMTEngine(delay_ms=5.0)


@pytest.fixture
def mock_ui_renderer() -> "MockUIRenderer":
    """Pre-configured MockUIRenderer."""
    from translator.infrastructure.mock.mock_ui import MockUIRenderer

    return MockUIRenderer(show_original=True)

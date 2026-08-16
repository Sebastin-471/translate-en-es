"""Tests for core event dataclasses."""

from __future__ import annotations

import time

from translator.core.events import (
    AudioChunk,
    PipelineShutdown,
    TranscriptResult,
    TranslationResult,
    VADSegment,
)


class TestAudioChunk:
    """Tests for AudioChunk dataclass."""

    def test_creation_with_defaults(self) -> None:
        chunk = AudioChunk(
            data=b"\x00\x00" * 480,
            sample_rate=16000,
            channels=1,
            duration_ms=30.0,
        )
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1
        assert chunk.duration_ms == 30.0
        assert len(chunk.data) == 960
        assert chunk.sequence_id  # auto-generated
        assert chunk.created_at_ns > 0

    def test_immutability(self) -> None:
        chunk = AudioChunk(data=b"\x00", sample_rate=16000, channels=1, duration_ms=1.0)
        try:
            chunk.sample_rate = 44100  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass

    def test_unique_sequence_ids(self) -> None:
        c1 = AudioChunk(data=b"", sample_rate=16000, channels=1, duration_ms=1.0)
        c2 = AudioChunk(data=b"", sample_rate=16000, channels=1, duration_ms=1.0)
        assert c1.sequence_id != c2.sequence_id

    def test_monotonic_timestamps(self) -> None:
        c1 = AudioChunk(data=b"", sample_rate=16000, channels=1, duration_ms=1.0)
        c2 = AudioChunk(data=b"", sample_rate=16000, channels=1, duration_ms=1.0)
        assert c2.created_at_ns >= c1.created_at_ns


class TestVADSegment:
    """Tests for VADSegment dataclass."""

    def test_creation(self) -> None:
        seg = VADSegment(
            audio_data=b"\x00" * 32000,
            sample_rate=16000,
            start_time_ms=100.0,
            end_time_ms=1100.0,
            duration_ms=1000.0,
            confidence=0.95,
        )
        assert seg.duration_ms == 1000.0
        assert seg.confidence == 0.95
        assert seg.sequence_id


class TestTranscriptResult:
    """Tests for TranscriptResult dataclass."""

    def test_creation(self) -> None:
        result = TranscriptResult(
            text="Hello world",
            language="en",
            confidence=0.88,
            segment_start_ms=0.0,
            segment_end_ms=1000.0,
            processing_time_ms=120.0,
        )
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.processing_time_ms == 120.0


class TestTranslationResult:
    """Tests for TranslationResult dataclass."""

    def test_creation(self) -> None:
        result = TranslationResult(
            original_text="Hello",
            translated_text="Hola",
            source_language="en",
            target_language="es",
            segment_start_ms=0.0,
            segment_end_ms=500.0,
            processing_time_ms=30.0,
        )
        assert result.original_text == "Hello"
        assert result.translated_text == "Hola"
        assert result.source_language == "en"
        assert result.target_language == "es"


class TestPipelineShutdown:
    """Tests for PipelineShutdown sentinel."""

    def test_default_reason(self) -> None:
        shutdown = PipelineShutdown()
        assert shutdown.reason == "shutdown"

    def test_custom_reason(self) -> None:
        shutdown = PipelineShutdown(reason="user_stop")
        assert shutdown.reason == "user_stop"

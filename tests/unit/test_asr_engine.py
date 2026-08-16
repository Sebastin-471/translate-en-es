"""Tests for the mock ASR engine."""

from __future__ import annotations

import pytest

from translator.core.events import VADSegment
from translator.infrastructure.mock.mock_asr import MockASREngine


class TestMockASREngine:
    """Tests for MockASREngine behavior."""

    @pytest.mark.asyncio
    async def test_returns_transcript(self) -> None:
        """ASR should return a TranscriptResult with text."""
        asr = MockASREngine(delay_ms=1, language="en")

        segment = VADSegment(
            audio_data=b"\x00" * 32000,
            sample_rate=16000,
            start_time_ms=0.0,
            end_time_ms=1000.0,
            duration_ms=1000.0,
            confidence=0.95,
        )

        result = await asr.transcribe(segment)
        assert result.text  # Non-empty text
        assert result.language == "en"
        assert result.confidence > 0
        assert result.sequence_id == segment.sequence_id

    @pytest.mark.asyncio
    async def test_rotates_texts(self) -> None:
        """ASR should cycle through sample texts."""
        asr = MockASREngine(delay_ms=1)

        segment = VADSegment(
            audio_data=b"\x00" * 32000,
            sample_rate=16000,
            start_time_ms=0.0,
            end_time_ms=1000.0,
            duration_ms=1000.0,
            confidence=0.95,
        )

        texts = set()
        for _ in range(10):
            result = await asr.transcribe(segment)
            texts.add(result.text)

        # Should have seen multiple different texts
        assert len(texts) > 1

    @pytest.mark.asyncio
    async def test_custom_texts(self) -> None:
        """ASR should use custom texts when provided."""
        custom = ["First sentence.", "Second sentence."]
        asr = MockASREngine(delay_ms=1, custom_texts=custom)

        segment = VADSegment(
            audio_data=b"\x00" * 32000,
            sample_rate=16000,
            start_time_ms=0.0,
            end_time_ms=1000.0,
            duration_ms=1000.0,
            confidence=0.95,
        )

        r1 = await asr.transcribe(segment)
        r2 = await asr.transcribe(segment)
        assert r1.text == "First sentence."
        assert r2.text == "Second sentence."

    @pytest.mark.asyncio
    async def test_preserves_sequence_id(self) -> None:
        """Transcript should carry the sequence_id from the VAD segment."""
        asr = MockASREngine(delay_ms=1)

        segment = VADSegment(
            audio_data=b"\x00" * 32000,
            sample_rate=16000,
            start_time_ms=0.0,
            end_time_ms=1000.0,
            duration_ms=1000.0,
            confidence=0.95,
            sequence_id="test-seq-123",
        )

        result = await asr.transcribe(segment)
        assert result.sequence_id == "test-seq-123"

    @pytest.mark.asyncio
    async def test_measures_processing_time(self) -> None:
        """Transcript should report processing_time_ms > 0."""
        asr = MockASREngine(delay_ms=10)

        segment = VADSegment(
            audio_data=b"\x00" * 32000,
            sample_rate=16000,
            start_time_ms=0.0,
            end_time_ms=1000.0,
            duration_ms=1000.0,
            confidence=0.95,
        )

        result = await asr.transcribe(segment)
        assert result.processing_time_ms >= 5  # At least some of the delay

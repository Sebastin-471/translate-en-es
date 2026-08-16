"""Tests for the mock MT engine."""

from __future__ import annotations

import pytest

from translator.core.events import TranscriptResult
from translator.infrastructure.mock.mock_mt import MockMTEngine


class TestMockMTEngine:
    """Tests for MockMTEngine behavior."""

    @pytest.mark.asyncio
    async def test_translates_known_text(self) -> None:
        """MT should return a known translation for predefined texts."""
        mt = MockMTEngine(delay_ms=1)

        transcript = TranscriptResult(
            text="Hello, how are you today?",
            language="en",
            confidence=0.9,
            segment_start_ms=0.0,
            segment_end_ms=1000.0,
            processing_time_ms=100.0,
        )

        result = await mt.translate(transcript)
        assert result.translated_text == "Hola, ¿cómo estás hoy?"
        assert result.source_language == "en"
        assert result.target_language == "es"
        assert result.original_text == "Hello, how are you today?"

    @pytest.mark.asyncio
    async def test_unknown_text_returns_default(self) -> None:
        """MT should return a default string for unknown texts."""
        mt = MockMTEngine(delay_ms=1)

        transcript = TranscriptResult(
            text="This is an unknown sentence.",
            language="en",
            confidence=0.9,
            segment_start_ms=0.0,
            segment_end_ms=500.0,
            processing_time_ms=50.0,
        )

        result = await mt.translate(transcript)
        assert result.translated_text == "[Traducción simulada]"

    @pytest.mark.asyncio
    async def test_custom_translations(self) -> None:
        """MT should use custom translations when provided."""
        custom = {"Good morning": "Buenos días"}
        mt = MockMTEngine(delay_ms=1, translations=custom)

        transcript = TranscriptResult(
            text="Good morning",
            language="en",
            confidence=0.9,
            segment_start_ms=0.0,
            segment_end_ms=500.0,
            processing_time_ms=50.0,
        )

        result = await mt.translate(transcript)
        assert result.translated_text == "Buenos días"

    @pytest.mark.asyncio
    async def test_preserves_sequence_id(self) -> None:
        """Translation should carry the sequence_id from the transcript."""
        mt = MockMTEngine(delay_ms=1)

        transcript = TranscriptResult(
            text="Hello, how are you today?",
            language="en",
            confidence=0.9,
            segment_start_ms=0.0,
            segment_end_ms=1000.0,
            processing_time_ms=100.0,
            sequence_id="seq-abc-123",
        )

        result = await mt.translate(transcript)
        assert result.sequence_id == "seq-abc-123"

    @pytest.mark.asyncio
    async def test_configurable_languages(self) -> None:
        """MT should report configured source/target languages."""
        mt = MockMTEngine(
            delay_ms=1,
            source_language="fr",
            target_language="de",
        )

        transcript = TranscriptResult(
            text="Bonjour",
            language="fr",
            confidence=0.9,
            segment_start_ms=0.0,
            segment_end_ms=500.0,
            processing_time_ms=50.0,
        )

        result = await mt.translate(transcript)
        assert result.source_language == "fr"
        assert result.target_language == "de"

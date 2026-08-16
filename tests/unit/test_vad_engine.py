"""Tests for the mock VAD engine."""

from __future__ import annotations

import pytest

from translator.core.events import AudioChunk
from translator.infrastructure.mock.mock_vad import MockVADEngine


class TestMockVADEngine:
    """Tests for MockVADEngine behavior."""

    @pytest.mark.asyncio
    async def test_emits_segment_after_n_chunks(self) -> None:
        """VAD should emit a segment after chunks_per_segment chunks."""
        vad = MockVADEngine(chunks_per_segment=3, delay_ms=0)

        chunk = AudioChunk(
            data=b"\x00\x00" * 480,
            sample_rate=16000,
            channels=1,
            duration_ms=30.0,
        )

        # First two chunks: no segment
        result1 = await vad.process_chunk(chunk)
        assert result1 is None

        result2 = await vad.process_chunk(chunk)
        assert result2 is None

        # Third chunk: segment emitted
        result3 = await vad.process_chunk(chunk)
        assert result3 is not None
        assert result3.sample_rate == 16000
        assert result3.confidence == 0.95
        assert len(result3.audio_data) == 3 * len(chunk.data)

    @pytest.mark.asyncio
    async def test_resets_after_segment(self) -> None:
        """After emitting a segment, the VAD should reset its buffer."""
        vad = MockVADEngine(chunks_per_segment=2, delay_ms=0)

        chunk = AudioChunk(
            data=b"\x00\x00" * 480,
            sample_rate=16000,
            channels=1,
            duration_ms=30.0,
        )

        # Emit first segment
        await vad.process_chunk(chunk)
        result = await vad.process_chunk(chunk)
        assert result is not None

        # Next chunk should start a new buffer
        result = await vad.process_chunk(chunk)
        assert result is None  # Not enough chunks yet

    @pytest.mark.asyncio
    async def test_reset_clears_state(self) -> None:
        """reset() should clear accumulated chunks."""
        vad = MockVADEngine(chunks_per_segment=3, delay_ms=0)

        chunk = AudioChunk(
            data=b"\x00\x00" * 480,
            sample_rate=16000,
            channels=1,
            duration_ms=30.0,
        )

        await vad.process_chunk(chunk)
        await vad.process_chunk(chunk)
        await vad.reset()

        # Should need 3 more chunks after reset
        result = await vad.process_chunk(chunk)
        assert result is None

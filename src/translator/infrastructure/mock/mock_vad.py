"""MockVADEngine: Simulated VAD that emits segments at fixed intervals.

Collects a configurable number of AudioChunks and then emits a
VADSegment, simulating speech detection without a real model.
"""

from __future__ import annotations

import asyncio

from translator.core.events import AudioChunk, VADSegment


class MockVADEngine:
    """VADEngine mock for UI development and CI testing.

    Accumulates a fixed number of chunks (simulating ~2 seconds of speech)
    and emits a VADSegment. Applies an artificial delay to simulate
    processing time.

    This class satisfies the VADEngine Protocol (structural subtyping).
    """

    def __init__(
        self,
        chunks_per_segment: int = 60,
        delay_ms: float = 5.0,
    ) -> None:
        """Initialize the mock VAD.

        Args:
            chunks_per_segment: Number of chunks to accumulate before
                emitting a segment (~60 chunks × 30ms = ~1.8s of audio).
            delay_ms: Artificial processing delay per chunk in ms.
        """
        self._chunks_per_segment = chunks_per_segment
        self._delay_ms = delay_ms
        self._buffer: list[bytes] = []
        self._chunk_count = 0
        self._elapsed_ms: float = 0.0

    async def process_chunk(self, chunk: AudioChunk) -> VADSegment | None:
        """Accumulate chunks and emit a segment every N chunks."""
        await asyncio.sleep(self._delay_ms / 1000.0)

        self._buffer.append(chunk.data)
        self._chunk_count += 1
        self._elapsed_ms += chunk.duration_ms

        if self._chunk_count >= self._chunks_per_segment:
            audio_data = b"".join(self._buffer)
            start_ms = self._elapsed_ms - (self._chunk_count * chunk.duration_ms)

            segment = VADSegment(
                audio_data=audio_data,
                sample_rate=chunk.sample_rate,
                start_time_ms=start_ms,
                end_time_ms=self._elapsed_ms,
                duration_ms=float(self._chunk_count * chunk.duration_ms),
                confidence=0.95,
            )

            self._buffer = []
            self._chunk_count = 0
            return segment

        return None

    async def reset(self) -> None:
        """Reset internal state."""
        self._buffer = []
        self._chunk_count = 0

"""MockASREngine: Simulated ASR that returns fixed/rotating text.

Returns pre-configured transcript texts with artificial delay
to simulate model inference time.
"""

from __future__ import annotations

import asyncio
import time

from translator.core.events import TranscriptResult, VADSegment


_SAMPLE_TRANSCRIPTS = [
    "Hello, how are you today?",
    "The meeting will start in five minutes.",
    "Can you share your screen please?",
    "I think we should discuss the next steps.",
    "Let me know if you have any questions.",
    "Thank you for your presentation.",
    "Could you repeat that last point?",
    "I agree with what was just said.",
]


class MockASREngine:
    """ASREngine mock for UI development and CI testing.

    Returns transcripts from a rotating list of sample sentences
    with configurable artificial delay.

    This class satisfies the ASREngine Protocol (structural subtyping).
    """

    def __init__(
        self,
        delay_ms: float = 200.0,
        language: str = "en",
        custom_texts: list[str] | None = None,
    ) -> None:
        """Initialize the mock ASR.

        Args:
            delay_ms: Artificial processing delay in ms.
            language: Source language code to report.
            custom_texts: Optional list of custom transcript texts.
        """
        self._delay_ms = delay_ms
        self._language = language
        self._texts = custom_texts or _SAMPLE_TRANSCRIPTS
        self._index = 0

    async def transcribe(self, segment: VADSegment) -> TranscriptResult:
        """Return a mock transcript after an artificial delay."""
        start_ns = time.monotonic_ns()

        await asyncio.sleep(self._delay_ms / 1000.0)

        text = self._texts[self._index % len(self._texts)]
        self._index += 1

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000

        return TranscriptResult(
            text=text,
            language=self._language,
            confidence=0.92,
            segment_start_ms=segment.start_time_ms,
            segment_end_ms=segment.end_time_ms,
            processing_time_ms=elapsed_ms,
            sequence_id=segment.sequence_id,
        )

    # --- ModelLifecycle (optional, for consistency) ---

    async def load_model(self) -> None:
        """No-op for mock."""

    async def unload_model(self) -> None:
        """No-op for mock."""

    def is_loaded(self) -> bool:
        """Always returns True for mock."""
        return True

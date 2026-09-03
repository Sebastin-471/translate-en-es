"""MockMTEngine: Simulated translation that returns fixed translations.

Returns pre-configured translations with artificial delay to simulate
model inference time. Useful for UI development and CI.
"""

from __future__ import annotations

import asyncio
import time

from translator.core.events import TranscriptResult, TranslationResult

_SAMPLE_TRANSLATIONS: dict[str, str] = {
    "Hello, how are you today?": "Hola, ¿cómo estás hoy?",
    "The meeting will start in five minutes.": "La reunión comenzará en cinco minutos.",
    "Can you share your screen please?": "¿Puedes compartir tu pantalla, por favor?",
    "I think we should discuss the next steps.": "Creo que deberíamos discutir los próximos pasos.",
    "Let me know if you have any questions.": "Avísame si tienes alguna pregunta.",
    "Thank you for your presentation.": "Gracias por tu presentación.",
    "Could you repeat that last point?": "¿Podrías repetir ese último punto?",
    "I agree with what was just said.": "Estoy de acuerdo con lo que se acaba de decir.",
}

_DEFAULT_TRANSLATION = "[Traducción simulada]"


class MockMTEngine:
    """MTEngine mock for UI development and CI testing.

    Looks up translations in a pre-configured dictionary, falling back
    to a default string for unknown inputs.

    This class satisfies the MTEngine Protocol (structural subtyping).
    """

    def __init__(
        self,
        delay_ms: float = 50.0,
        source_language: str = "en",
        target_language: str = "es",
        translations: dict[str, str] | None = None,
    ) -> None:
        """Initialize the mock MT.

        Args:
            delay_ms: Artificial processing delay in ms.
            source_language: Source language code.
            target_language: Target language code.
            translations: Optional custom translation dictionary.
        """
        self._delay_ms = delay_ms
        self._source_language = source_language
        self._target_language = target_language
        self._translations = translations or _SAMPLE_TRANSLATIONS

    async def translate(self, transcript: TranscriptResult) -> TranslationResult:
        """Return a mock translation after an artificial delay."""
        start_ns = time.monotonic_ns()

        await asyncio.sleep(self._delay_ms / 1000.0)

        translated = self._translations.get(transcript.text, _DEFAULT_TRANSLATION)
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000

        return TranslationResult(
            original_text=transcript.text,
            translated_text=translated,
            source_language=self._source_language,
            target_language=self._target_language,
            segment_start_ms=transcript.segment_start_ms,
            segment_end_ms=transcript.segment_end_ms,
            processing_time_ms=elapsed_ms,
            sequence_id=transcript.sequence_id,
        )

    # --- ModelLifecycle (optional, for consistency) ---

    async def load_model(self) -> None:
        """No-op for mock."""

    async def unload_model(self) -> None:
        """No-op for mock."""

    def is_loaded(self) -> bool:
        """Always returns True for mock."""
        return True

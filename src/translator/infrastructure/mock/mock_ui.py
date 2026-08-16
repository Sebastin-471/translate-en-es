"""MockUIRenderer: Logs translations to console instead of rendering a GUI.

Used for headless testing, CI, and development without a display server.
"""

from __future__ import annotations

import structlog

from translator.core.events import TranslationResult

logger = structlog.get_logger(__name__)


class MockUIRenderer:
    """UIRenderer mock that logs translations to structured logger.

    Useful for testing the full pipeline without requiring a GUI
    environment (e.g., in CI or SSH sessions).

    This class satisfies the UIRenderer Protocol (structural subtyping).
    """

    def __init__(self, show_original: bool = True) -> None:
        self._show_original = show_original
        self._translations: list[TranslationResult] = []
        self._running = False

    async def show(self, translation: TranslationResult) -> None:
        """Log the translation to the structured logger."""
        self._translations.append(translation)

        if self._show_original:
            logger.info(
                "ui_subtitle",
                original=translation.original_text,
                translated=translation.translated_text,
                source_lang=translation.source_language,
                target_lang=translation.target_language,
                sequence_id=translation.sequence_id,
            )
        else:
            logger.info(
                "ui_subtitle",
                translated=translation.translated_text,
                target_lang=translation.target_language,
                sequence_id=translation.sequence_id,
            )

    async def clear(self) -> None:
        """Clear the translation history."""
        self._translations = []
        logger.debug("ui_cleared")

    async def start(self) -> None:
        """Mark as running."""
        self._running = True
        logger.info("mock_ui_started")

    async def stop(self) -> None:
        """Mark as stopped."""
        self._running = False
        logger.info("mock_ui_stopped", total_translations=len(self._translations))

    @property
    def translation_history(self) -> list[TranslationResult]:
        """Access recorded translations (useful in tests)."""
        return list(self._translations)

"""MarianMTEngine: Machine translation using MarianMT via CTranslate2.

Receives TranscriptResults and produces TranslationResults. The source
and target languages are fully configurable — no hardcoded language codes.

Model conversion flow:
  1. First run: downloads the HuggingFace MarianMT model.
  2. Converts it to CTranslate2 format and caches locally.
  3. Subsequent runs: loads from the cached CTranslate2 directory.

Requirements:
  pip install ctranslate2 transformers sentencepiece
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from translator.core.events import TranscriptResult, TranslationResult

if TYPE_CHECKING:
    from translator.core.config import MTConfig

logger = structlog.get_logger(__name__)

# Default cache directory for converted models
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "translate-en-es" / "models"


class MarianMTEngine:
    """MTEngine implementation using MarianMT via CTranslate2.

    Uses the HuggingFace tokenizer for pre/post-processing and CTranslate2
    for optimized inference. Supports both CPU and CUDA.

    This class satisfies both the MTEngine and ModelLifecycle Protocols.
    """

    def __init__(self, config: MTConfig) -> None:
        self._config = config
        self._translator: Any = None  # ctranslate2.Translator
        self._tokenizer: Any = None  # transformers.AutoTokenizer
        self._loaded = False

    async def translate(self, transcript: TranscriptResult) -> TranslationResult:
        """Translate a transcript from source to target language."""
        if not self._loaded:
            await self.load_model()

        if not transcript.text.strip():
            return TranslationResult(
                original_text=transcript.text,
                translated_text="",
                source_language=self._config.source_language,
                target_language=self._config.target_language,
                segment_start_ms=transcript.segment_start_ms,
                segment_end_ms=transcript.segment_end_ms,
                processing_time_ms=0.0,
                sequence_id=transcript.sequence_id,
            )

        start_ns = time.monotonic_ns()

        # Tokenize input
        encoded = self._tokenizer.encode(transcript.text)
        tokens = self._tokenizer.convert_ids_to_tokens(encoded)

        # Translate via CTranslate2
        results = self._translator.translate_batch(
            [tokens],
            beam_size=self._config.beam_size,
        )

        # Decode output
        output_tokens = results[0].hypotheses[0]
        output_ids = self._tokenizer.convert_tokens_to_ids(output_tokens)
        translated_text = self._tokenizer.decode(output_ids, skip_special_tokens=True)

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000

        result = TranslationResult(
            original_text=transcript.text,
            translated_text=translated_text,
            source_language=self._config.source_language,
            target_language=self._config.target_language,
            segment_start_ms=transcript.segment_start_ms,
            segment_end_ms=transcript.segment_end_ms,
            processing_time_ms=elapsed_ms,
            sequence_id=transcript.sequence_id,
        )

        logger.info(
            "mt_translation_complete",
            original=transcript.text[:80],
            translated=translated_text[:80],
            source_lang=self._config.source_language,
            target_lang=self._config.target_language,
            processing_ms=round(elapsed_ms, 1),
            sequence_id=result.sequence_id,
        )

        return result

    # --- ModelLifecycle ---

    async def load_model(self) -> None:
        """Load the MarianMT model (converting if needed)."""
        try:
            import ctranslate2  # type: ignore[import-untyped]
            import transformers  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "ctranslate2 and transformers are required. "
                "Install with: pip install ctranslate2 transformers sentencepiece"
            ) from e

        model_name = self._config.model_name
        ct2_dir = self._get_ct2_model_dir()

        # Load tokenizer
        logger.info("mt_tokenizer_loading", model=model_name)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        # Convert to CTranslate2 format if not cached
        if not ct2_dir.exists() or not (ct2_dir / "model.bin").exists():
            logger.info("mt_model_converting", model=model_name, output_dir=str(ct2_dir))
            ct2_dir.mkdir(parents=True, exist_ok=True)
            converter = ctranslate2.converters.TransformersConverter(model_name)
            converter.convert(
                str(ct2_dir),
                quantization=self._config.compute_type,
                force=True,
            )
            logger.info("mt_model_converted", output_dir=str(ct2_dir))

        # Load CTranslate2 translator
        device = self._resolve_device()
        logger.info(
            "mt_model_loading",
            model_dir=str(ct2_dir),
            device=device,
            compute_type=self._config.compute_type,
        )

        start = time.monotonic()
        self._translator = ctranslate2.Translator(
            str(ct2_dir),
            device=device,
            compute_type=self._config.compute_type,
        )
        elapsed = time.monotonic() - start

        self._loaded = True
        logger.info(
            "mt_model_loaded",
            model=model_name,
            device=device,
            load_time_s=round(elapsed, 2),
        )

    async def unload_model(self) -> None:
        """Unload the model and free resources."""
        self._translator = None
        self._tokenizer = None
        self._loaded = False
        logger.info("mt_model_unloaded")

    def is_loaded(self) -> bool:
        """Return True if the model is currently loaded."""
        return self._loaded

    # --- Private helpers ---

    def _get_ct2_model_dir(self) -> Path:
        """Get the directory for the cached CTranslate2 model."""
        if self._config.model_path:
            return Path(self._config.model_path)

        # Derive directory name from model name
        # e.g., "Helsinki-NLP/opus-mt-en-es" → "opus-mt-en-es-ct2"
        model_basename = self._config.model_name.split("/")[-1]
        return _DEFAULT_CACHE_DIR / f"{model_basename}-ct2"

    def _resolve_device(self) -> str:
        """Resolve 'auto' device to 'cuda' or 'cpu'."""
        if self._config.device != "auto":
            return self._config.device

        try:
            import ctranslate2  # type: ignore[import-untyped]

            if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
                return "cuda"
        except Exception:
            pass
        return "cpu"

"""WhisperASREngine: Speech-to-text using faster-whisper / CTranslate2.

Receives VADSegments and produces TranscriptResults using the
faster-whisper library for optimized Whisper inference.

Requirements:
  pip install faster-whisper
  For GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import structlog

from translator.core.events import TranscriptResult, VADSegment
from translator.core.model_manager import DeviceType, ModelManager

if TYPE_CHECKING:
    from translator.core.config import ASRConfig

logger = structlog.get_logger(__name__)


class WhisperASREngine:
    """ASREngine implementation using faster-whisper (CTranslate2 backend).

    Supports multiple Whisper model sizes and automatic device selection
    (CUDA/CPU). Model size and compute type are fully configurable via
    ASRConfig — no hardcoded values.

    This class satisfies both the ASREngine and ModelLifecycle Protocols.
    """

    def __init__(self, config: ASRConfig, model_manager: ModelManager) -> None:
        self._config = config
        self._model_manager = model_manager
        self._model_name = "whisper"
        self._model: Any = None
        self._loaded = False

    async def transcribe(self, segment: VADSegment) -> TranscriptResult:
        """Transcribe a VAD segment to text using faster-whisper."""
        if not self._loaded:
            await self.load_model()

        start_ns = time.monotonic_ns()

        # Convert PCM bytes to WAV in-memory for faster-whisper
        audio_array = self._pcm_to_numpy(segment.audio_data, segment.sample_rate)

        # Run transcription
        segments_iter, info = await asyncio.to_thread(
            self._model.transcribe,
            audio_array,
            language=self._config.language,
            beam_size=self._config.beam_size,
            vad_filter=False,  # We already ran VAD
            condition_on_previous_text=False,  # Prevents hallucinations on VAD-chunked audio
        )

        # Collect all segment texts
        texts: list[str] = []
        for seg in segments_iter:
            texts.append(seg.text.strip())

        full_text = " ".join(texts)
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000

        result = TranscriptResult(
            text=full_text,
            language=info.language if info.language else self._config.language,
            confidence=info.language_probability if info.language_probability else 0.0,
            segment_start_ms=segment.start_time_ms,
            segment_end_ms=segment.end_time_ms,
            processing_time_ms=elapsed_ms,
            is_partial=segment.is_partial,
            sequence_id=segment.sequence_id,
        )

        logger.info(
            "asr_transcription_complete",
            text=full_text[:100],
            language=result.language,
            confidence=result.confidence,
            processing_ms=round(elapsed_ms, 1),
            sequence_id=result.sequence_id,
        )

        return result

    # --- ModelLifecycle ---

    async def load_model(self) -> None:
        """Load the faster-whisper model."""
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "faster-whisper is required. Install with: pip install faster-whisper"
            ) from e

        device_str = self._resolve_device()
        dev_type = DeviceType.CUDA if device_str == "cuda" else DeviceType.CPU
        model_path = self._config.model_path or self._config.model_size

        # Estimate VRAM
        vram_estimate = 1500 if "large" in model_path else (500 if "small" in model_path else 200)

        logger.info(
            "asr_model_loading",
            model=model_path,
            device=device_str,
            compute_type=self._config.compute_type,
        )

        await self._model_manager.register_model(self._model_name, "asr", vram_estimate)

        def _loader() -> Any:
            return WhisperModel(
                model_path,
                device=device_str,
                compute_type=self._config.compute_type,
                cpu_threads=self._config.cpu_threads,
            )

        start = time.monotonic()
        self._model = await self._model_manager.load_model(self._model_name, _loader, dev_type)
        elapsed = time.monotonic() - start

        self._loaded = True
        logger.info(
            "asr_model_loaded",
            model=model_path,
            device=device_str,
            load_time_s=round(elapsed, 2),
        )

    async def unload_model(self) -> None:
        """Unload the model and free resources."""
        if self._loaded:
            await self._model_manager.unload_model(self._model_name)
        self._model = None
        self._loaded = False
        logger.info("asr_model_unloaded")

    def is_loaded(self) -> bool:
        """Return True if the model is currently loaded."""
        return self._loaded

    # --- Private helpers ---

    def _resolve_device(self) -> str:
        """Resolve 'auto' device to 'cuda' or 'cpu'."""
        if self._config.device != "auto":
            return self._config.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def _pcm_to_numpy(pcm_data: bytes, sample_rate: int) -> Any:
        """Convert 16-bit PCM bytes to numpy float32 array for faster-whisper."""
        import numpy as np

        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        return samples

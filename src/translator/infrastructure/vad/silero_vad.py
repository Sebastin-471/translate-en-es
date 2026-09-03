"""SileroVADEngine: Voice Activity Detection using Silero VAD v6.

Processes AudioChunks and accumulates speech frames into complete
VADSegments. Uses the silero-vad package for lightweight, CPU-efficient
speech detection.

Requirements:
  pip install silero-vad torch
"""

from __future__ import annotations

import struct
import time
import uuid
from typing import TYPE_CHECKING, Any

import structlog

from translator.core.events import AudioChunk, VADSegment
from translator.core.model_manager import DeviceType, ModelManager

if TYPE_CHECKING:
    from translator.core.config import VADConfig

logger = structlog.get_logger(__name__)


class SileroVADEngine:
    """VADEngine implementation using Silero VAD.

    Accumulates audio frames and detects speech regions based on
    configurable thresholds. Emits a VADSegment when a complete
    speech region (onset → offset + padding) is detected.

    This class satisfies the VADEngine Protocol (structural subtyping).
    """

    def __init__(self, config: VADConfig, model_manager: ModelManager, sample_rate: int = 16_000) -> None:
        self._config = config
        self._model_manager = model_manager
        self._model_name = "silero_vad"
        self._sample_rate = sample_rate
        self._model: Any = None

        # State for accumulating speech
        self._is_speaking = False
        self._speech_buffer: list[bytes] = []
        self._ring_buffer: list[bytes] = []
        self._speech_start_ms: float = 0.0
        self._silence_duration_ms: float = 0.0
        self._total_speech_ms: float = 0.0
        self._elapsed_ms: float = 0.0
        self._last_confidence: float = 0.0
        self._last_vad_log: float = 0.0
        self._confidence_sum: float = 0.0
        self._confidence_count: int = 0
        self._speech_detections: int = 0
        self._current_sequence_id: str = ""
        self._last_partial_ms: float = 0.0

    async def process_chunk(self, chunk: AudioChunk) -> VADSegment | None:
        """Process an AudioChunk and return a VADSegment if speech ended."""
        if self._model is None:
            await self._load_model()

        # Maintain a ring buffer for pre-speech padding
        if not self._is_speaking:
            self._ring_buffer.append(chunk.data)
            max_ring_chunks = max(1, self._config.speech_pad_ms // chunk.duration_ms)
            if len(self._ring_buffer) > max_ring_chunks:
                self._ring_buffer.pop(0)

        # Convert PCM bytes to float tensor for Silero
        samples = self._pcm_to_float(chunk.data)
        confidence = self._run_vad(samples)
        self._last_confidence = confidence
        self._elapsed_ms += chunk.duration_ms
        self._confidence_sum += confidence
        self._confidence_count += 1

        is_speech = confidence >= self._config.threshold
        if is_speech:
            self._speech_detections += 1

        # Periodic VAD stats logging (every 5 seconds)
        now = time.monotonic()
        if self._last_vad_log == 0.0:
            self._last_vad_log = now
        if now - self._last_vad_log >= 5.0:
            avg_conf = self._confidence_sum / max(self._confidence_count, 1)
            logger.info(
                "vad_stats",
                avg_confidence=round(avg_conf, 4),
                last_confidence=round(confidence, 4),
                speech_detections=self._speech_detections,
                total_chunks=self._confidence_count,
                threshold=self._config.threshold,
            )
            self._confidence_sum = 0.0
            self._confidence_count = 0
            self._speech_detections = 0
            self._last_vad_log = now

        if is_speech:
            if not self._is_speaking:
                # Speech onset
                self._is_speaking = True
                self._speech_start_ms = self._elapsed_ms - chunk.duration_ms - (len(self._ring_buffer) * chunk.duration_ms)
                self._speech_buffer = self._ring_buffer.copy()
                self._ring_buffer = []
                self._total_speech_ms = len(self._speech_buffer) * chunk.duration_ms
                self._silence_duration_ms = 0.0
                self._current_sequence_id = uuid.uuid4().hex[:12]
                self._last_partial_ms = 0.0
                logger.debug(
                    "vad_speech_start",
                    start_ms=self._speech_start_ms,
                    confidence=confidence,
                )

            self._speech_buffer.append(chunk.data)
            self._total_speech_ms += chunk.duration_ms
            self._silence_duration_ms = 0.0

            # Force split if segment is too long
            if self._total_speech_ms >= self._config.max_segment_duration_ms:
                return self._emit_segment(is_partial=False)

            # Emit partial segment if interval reached
            elif self._total_speech_ms - self._last_partial_ms >= getattr(self._config, "partial_interval_ms", 500):
                self._last_partial_ms = self._total_speech_ms
                return self._emit_segment(is_partial=True)

        else:
            if self._is_speaking:
                self._silence_duration_ms += chunk.duration_ms
                # Still add to buffer (padding)
                self._speech_buffer.append(chunk.data)
                self._total_speech_ms += chunk.duration_ms

                if self._silence_duration_ms >= self._config.min_silence_duration_ms:
                    # Speech ended — check minimum duration
                    speech_only_ms = self._total_speech_ms - self._silence_duration_ms
                    if speech_only_ms >= self._config.min_speech_duration_ms:
                        return self._emit_segment(is_partial=False)
                    else:
                        # Too short — discard
                        logger.debug(
                            "vad_segment_too_short",
                            duration_ms=speech_only_ms,
                            min_required_ms=self._config.min_speech_duration_ms,
                        )
                        self._reset_state()

        return None

    async def reset(self) -> None:
        """Reset internal VAD state."""
        self._reset_state()
        if self._model is not None:
            self._model.reset_states()

    # --- Private helpers ---

    async def _load_model(self) -> None:
        """Load the Silero VAD model."""
        try:
            from silero_vad import load_silero_vad  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "silero-vad is required. Install with: pip install silero-vad torch"
            ) from e

        # VAD uses negligible VRAM (~10MB)
        await self._model_manager.register_model(self._model_name, "vad", 10)

        def _loader() -> Any:
            return load_silero_vad()

        self._model = await self._model_manager.load_model(
            self._model_name, _loader, DeviceType.CPU
        )
        logger.info("vad_model_loaded", model="silero-vad-v6")

    def _run_vad(self, samples: list[float]) -> float:
        """Run VAD inference on a list of float samples."""
        import torch

        tensor = torch.FloatTensor(samples)
        confidence: float = self._model(tensor, self._sample_rate).item()
        return confidence

    def _emit_segment(self, is_partial: bool = False) -> VADSegment:
        """Create a VADSegment from accumulated speech buffer."""
        audio_data = b"".join(self._speech_buffer)
        segment = VADSegment(
            audio_data=audio_data,
            sample_rate=self._sample_rate,
            start_time_ms=self._speech_start_ms,
            end_time_ms=self._elapsed_ms,
            duration_ms=self._total_speech_ms,
            confidence=self._last_confidence,
            is_partial=is_partial,
            sequence_id=self._current_sequence_id or uuid.uuid4().hex[:12],
        )

        logger.info(
            "vad_segment_emitted",
            start_ms=segment.start_time_ms,
            end_ms=segment.end_time_ms,
            duration_ms=segment.duration_ms,
            confidence=segment.confidence,
            is_partial=segment.is_partial,
            sequence_id=segment.sequence_id,
        )

        if not is_partial:
            self._reset_state()
        return segment

    def _reset_state(self) -> None:
        """Reset accumulation state without resetting the model."""
        self._is_speaking = False
        self._speech_buffer = []
        self._ring_buffer = []
        self._speech_start_ms = 0.0
        self._silence_duration_ms = 0.0
        self._total_speech_ms = 0.0
        self._current_sequence_id = ""
        self._last_partial_ms = 0.0

    @staticmethod
    def _pcm_to_float(data: bytes) -> list[float]:
        """Convert 16-bit PCM bytes to normalized float samples [-1.0, 1.0]."""
        n_samples = len(data) // 2
        samples = struct.unpack(f"<{n_samples}h", data)
        return [s / 32768.0 for s in samples]

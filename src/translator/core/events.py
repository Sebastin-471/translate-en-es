"""Typed message dataclasses for inter-stage communication.

Every message flowing through the pipeline is an instance of one of these
frozen dataclasses. They are the ONLY data structures that cross queue
boundaries between stages.

Design notes:
  - All dataclasses are frozen (immutable) to prevent shared mutable state.
  - Each carries a `created_at` timestamp (monotonic) for latency measurement.
  - `sequence_id` provides end-to-end tracing of a single audio segment
    across all pipeline stages.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


def _monotonic_ns() -> int:
    """Return current monotonic time in nanoseconds for latency tracking."""
    return time.monotonic_ns()


def _new_sequence_id() -> str:
    """Generate a unique sequence ID for tracing a segment across stages."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Stage 1 → 2: AudioSource → VADEngine
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A raw audio chunk captured from the system audio source.

    Attributes:
        data: Raw PCM bytes (16-bit signed LE mono).
        sample_rate: Sample rate in Hz (expected: 16000).
        channels: Number of audio channels (expected: 1).
        duration_ms: Duration of this chunk in milliseconds.
        sequence_id: Unique ID for tracing this chunk through the pipeline.
        created_at_ns: Monotonic timestamp (ns) when this chunk was created.
    """

    data: bytes
    sample_rate: int
    channels: int
    duration_ms: float
    sequence_id: str = field(default_factory=_new_sequence_id)
    created_at_ns: int = field(default_factory=_monotonic_ns)


# ---------------------------------------------------------------------------
# Stage 2 → 3: VADEngine → ASREngine
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VADSegment:
    """A segment of audio detected as containing speech by the VAD.

    The VAD accumulates AudioChunks and emits a VADSegment when it detects
    a complete speech region (speech start → silence end with padding).

    Attributes:
        audio_data: Concatenated raw PCM bytes of the speech segment.
        sample_rate: Sample rate in Hz.
        start_time_ms: Start of speech relative to the capture session.
        end_time_ms: End of speech relative to the capture session.
        duration_ms: Duration of the speech segment in milliseconds.
        confidence: VAD confidence score (0.0–1.0).
        sequence_id: Inherited from the first AudioChunk in this segment.
        created_at_ns: Monotonic timestamp (ns) when this segment was created.
    """

    audio_data: bytes
    sample_rate: int
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    confidence: float
    sequence_id: str = field(default_factory=_new_sequence_id)
    created_at_ns: int = field(default_factory=_monotonic_ns)


# ---------------------------------------------------------------------------
# Stage 3 → 4: ASREngine → MTEngine
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    """The output of speech-to-text (ASR) for a single VAD segment.

    Attributes:
        text: Recognized text in the source language.
        language: Detected or configured source language (ISO 639-1).
        confidence: ASR confidence score (0.0–1.0), if available.
        segment_start_ms: Start time of the original speech segment.
        segment_end_ms: End time of the original speech segment.
        processing_time_ms: Time spent in the ASR engine for this segment.
        sequence_id: Inherited from the VADSegment.
        created_at_ns: Monotonic timestamp (ns) when this result was created.
    """

    text: str
    language: str
    confidence: float
    segment_start_ms: float
    segment_end_ms: float
    processing_time_ms: float
    sequence_id: str = field(default_factory=_new_sequence_id)
    created_at_ns: int = field(default_factory=_monotonic_ns)


# ---------------------------------------------------------------------------
# Stage 4 → 5: MTEngine → UIRenderer
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """The output of machine translation for a single transcript.

    Attributes:
        original_text: The source-language text (from ASR).
        translated_text: The target-language translation.
        source_language: Source language (ISO 639-1).
        target_language: Target language (ISO 639-1).
        segment_start_ms: Start time of the original speech segment.
        segment_end_ms: End time of the original speech segment.
        processing_time_ms: Time spent in the MT engine for this segment.
        sequence_id: Inherited from the TranscriptResult.
        created_at_ns: Monotonic timestamp (ns) when this result was created.
    """

    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    segment_start_ms: float
    segment_end_ms: float
    processing_time_ms: float
    sequence_id: str = field(default_factory=_new_sequence_id)
    created_at_ns: int = field(default_factory=_monotonic_ns)


# ---------------------------------------------------------------------------
# Sentinel for graceful shutdown
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineShutdown:
    """Sentinel message to signal graceful shutdown of a pipeline stage.

    When a stage receives this on its input queue, it should:
    1. Finish processing any in-flight work.
    2. Forward the sentinel to its output queue.
    3. Return / stop its async loop.
    """

    reason: str = "shutdown"

"""Core interface definitions (Protocols) for all pipeline stages.

These Protocols define the contracts that concrete implementations in
infrastructure/ must satisfy. They use `typing.Protocol` (structural
subtyping) rather than `abc.ABC` so that:
  1. Implementations don't need to inherit — any class with matching
     method signatures satisfies the protocol (duck typing + static checks).
  2. Third-party code can be wrapped without forcing it into an inheritance
     hierarchy.

Import rules:
  - This module may only import from `translator.core.events` and stdlib.
  - No infrastructure, pipeline, or UI imports allowed here.

All async methods use `async def` because every stage runs in an asyncio
event loop and may need to perform I/O (audio capture, model inference,
file reads, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:

    from translator.core.events import (
        AudioChunk,
        TranscriptResult,
        TranslationResult,
        VADSegment,
    )


# ---------------------------------------------------------------------------
# Stage 1: Audio Source
# ---------------------------------------------------------------------------


@runtime_checkable
class AudioSource(Protocol):
    """Captures system audio and produces AudioChunk messages.

    Implementations:
      - WASAPIAudioSource (Windows loopback)
      - PipeWireAudioSource (Linux PipeWire/PulseAudio monitor)
      - FileAudioSource (WAV file for testing)
      - MockAudioSource (synthetic data for dev/CI)

    Lifecycle:
      1. `start()` — open the audio device / file.
      2. `read_chunk()` — called repeatedly by the pipeline to get the next
         chunk. Should block (async) until data is available.
      3. `stop()` — release the audio device / close the file.
    """

    async def start(self) -> None:
        """Open the audio source and prepare for capture."""
        ...

    async def read_chunk(self) -> AudioChunk:
        """Read and return the next audio chunk.

        This method should block until a chunk of the configured duration
        is available. Returns an AudioChunk with raw PCM data.
        """
        ...

    async def stop(self) -> None:
        """Release audio resources and stop capture."""
        ...


# ---------------------------------------------------------------------------
# Stage 2: Voice Activity Detection
# ---------------------------------------------------------------------------


@runtime_checkable
class VADEngine(Protocol):
    """Detects speech regions in an audio stream.

    The VAD accumulates AudioChunks and emits a VADSegment when it
    detects a complete speech utterance (speech onset → silence offset,
    with configurable padding and duration thresholds).

    Implementations:
      - SileroVADEngine (Silero VAD v6)
      - MockVADEngine (always detects speech after N chunks)
    """

    async def process_chunk(self, chunk: AudioChunk) -> VADSegment | None:
        """Process a single audio chunk through the VAD.

        Args:
            chunk: Raw audio chunk from the AudioSource.

        Returns:
            A VADSegment if a complete speech region was detected,
            or None if the chunk is part of ongoing speech/silence.
        """
        ...

    async def reset(self) -> None:
        """Reset internal VAD state (e.g., between sessions)."""
        ...


# ---------------------------------------------------------------------------
# Stage 3: Automatic Speech Recognition
# ---------------------------------------------------------------------------


@runtime_checkable
class ASREngine(Protocol):
    """Transcribes a speech segment into text.

    Receives a VADSegment (a complete speech utterance) and returns
    a TranscriptResult with the recognized text.

    Implementations:
      - WhisperASREngine (faster-whisper / CTranslate2)
      - MockASREngine (returns fixed text after a delay)
    """

    async def transcribe(self, segment: VADSegment) -> TranscriptResult:
        """Transcribe a speech segment to text.

        Args:
            segment: A VAD segment containing speech audio.

        Returns:
            A TranscriptResult with the recognized text, language,
            and confidence score.
        """
        ...


# ---------------------------------------------------------------------------
# Stage 4: Machine Translation
# ---------------------------------------------------------------------------


@runtime_checkable
class MTEngine(Protocol):
    """Translates text from a source language to a target language.

    Receives a TranscriptResult and returns a TranslationResult.
    The source and target languages are configurable — no hardcoded
    language codes in the implementation.

    Implementations:
      - MarianMTEngine (MarianMT via CTranslate2)
      - MockMTEngine (returns fixed translation after a delay)
    """

    async def translate(self, transcript: TranscriptResult) -> TranslationResult:
        """Translate a transcript from source to target language.

        Args:
            transcript: The ASR output to translate.

        Returns:
            A TranslationResult with the original and translated text.
        """
        ...


# ---------------------------------------------------------------------------
# Stage 5: UI Renderer
# ---------------------------------------------------------------------------


@runtime_checkable
class UIRenderer(Protocol):
    """Renders translated subtitles to the user.

    The UI renderer receives TranslationResult events and displays them
    as an always-on-top overlay. It runs in its own thread (Tkinter
    requires the main thread on some platforms) and communicates with
    the pipeline via a thread-safe queue.

    Implementations:
      - TkinterOverlayRenderer (Tkinter always-on-top overlay)
      - MockUIRenderer (logs to console, no GUI)
    """

    async def show(self, translation: TranslationResult) -> None:
        """Display a translation result in the overlay.

        Args:
            translation: The translation to display.
        """
        ...

    async def clear(self) -> None:
        """Clear all currently displayed subtitles."""
        ...

    async def start(self) -> None:
        """Initialize and show the UI overlay."""
        ...

    async def stop(self) -> None:
        """Close the UI overlay and release resources."""
        ...


# ---------------------------------------------------------------------------
# Supporting Protocol: Model Lifecycle Hooks
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelLifecycle(Protocol):
    """Optional protocol for engines that manage heavy ML models.

    Engines that load large models (ASR, MT) can implement this to
    participate in centralized model management via the ModelManager.
    """

    async def load_model(self) -> None:
        """Load the ML model into memory (CPU/GPU)."""
        ...

    async def unload_model(self) -> None:
        """Unload the ML model and free resources."""
        ...

    def is_loaded(self) -> bool:
        """Return True if the model is currently loaded."""
        ...

"""MockAudioSource: Generates synthetic audio chunks for testing.

Produces silent (or optionally sine-wave) PCM audio at the configured
sample rate and chunk duration, with artificial delays to simulate
real-time capture.
"""

from __future__ import annotations

import asyncio
import math
import struct
from typing import TYPE_CHECKING

from translator.core.events import AudioChunk

if TYPE_CHECKING:
    from translator.core.config import AudioConfig


class MockAudioSource:
    """AudioSource mock for UI development and CI testing.

    Generates synthetic audio chunks without requiring any audio device.
    Configurable to produce silence or a sine wave tone.

    This class satisfies the AudioSource Protocol (structural subtyping).
    """

    def __init__(
        self,
        config: AudioConfig,
        generate_tone: bool = False,
        tone_freq: float = 440.0,
    ) -> None:
        self._config = config
        self._generate_tone = generate_tone
        self._tone_freq = tone_freq
        self._running = False
        self._chunk_count = 0
        self._elapsed_ms: float = 0.0

    async def start(self) -> None:
        """Start generating synthetic audio."""
        self._running = True
        self._chunk_count = 0
        self._elapsed_ms = 0.0

    async def read_chunk(self) -> AudioChunk:
        """Generate and return a synthetic audio chunk."""
        if not self._running:
            raise RuntimeError("MockAudioSource not started — call start() first")

        duration_s = self._config.chunk_duration_ms / 1000.0
        n_samples = int(self._config.sample_rate * duration_s)

        if self._generate_tone:
            data = self._generate_sine(n_samples, self._tone_freq, self._config.sample_rate)
        else:
            data = b"\x00" * (n_samples * 2)  # 16-bit silence

        self._chunk_count += 1
        self._elapsed_ms += self._config.chunk_duration_ms

        chunk = AudioChunk(
            data=data,
            sample_rate=self._config.sample_rate,
            channels=1,
            duration_ms=float(self._config.chunk_duration_ms),
        )

        # Simulate real-time delay
        await asyncio.sleep(duration_s)
        return chunk

    async def stop(self) -> None:
        """Stop generating audio."""
        self._running = False

    @staticmethod
    def _generate_sine(n_samples: int, freq: float, sample_rate: int) -> bytes:
        """Generate a sine wave as 16-bit PCM bytes."""
        samples: list[int] = []
        for i in range(n_samples):
            val = math.sin(2 * math.pi * freq * i / sample_rate)
            samples.append(int(val * 16000))  # ~50% amplitude
        return struct.pack(f"<{n_samples}h", *samples)

"""FileAudioSource: reads a WAV file and emits AudioChunks simulating real-time.

Used for testing, demos, and integration tests without requiring a live
audio device. Respects the configured chunk_duration_ms to simulate
real-time playback speed.
"""

from __future__ import annotations

import asyncio
import io
import wave
from typing import TYPE_CHECKING

from translator.core.events import AudioChunk

if TYPE_CHECKING:
    from translator.core.config import AudioConfig


class FileAudioSource:
    """AudioSource implementation that reads from a WAV file.

    Emits AudioChunk messages at real-time speed (based on chunk_duration_ms).
    Automatically resamples to 16kHz mono if the source file differs.

    This class satisfies the AudioSource Protocol without inheriting from it
    (structural subtyping via typing.Protocol).
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._wav: wave.Wave_read | None = None
        self._bytes_per_chunk: int = 0
        self._chunk_duration_s: float = 0.0
        self._finished = False
        self._elapsed_ms: float = 0.0

    async def start(self) -> None:
        """Open the WAV file and compute chunk sizes."""
        if not self._config.file_path:
            raise ValueError("FileAudioSource requires audio.file_path in config")

        self._wav = wave.open(self._config.file_path, "rb")
        wav = self._wav

        # Validate format: we expect 16-bit PCM
        if wav.getsampwidth() != 2:
            raise ValueError(
                f"WAV file must be 16-bit PCM, got {wav.getsampwidth() * 8}-bit"
            )

        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()

        # Compute bytes per chunk based on the file's actual sample rate
        self._chunk_duration_s = self._config.chunk_duration_ms / 1000.0
        frames_per_chunk = int(sample_rate * self._chunk_duration_s)
        bytes_per_frame = n_channels * wav.getsampwidth()
        self._bytes_per_chunk = frames_per_chunk * bytes_per_frame

        self._finished = False
        self._elapsed_ms = 0.0

    async def read_chunk(self) -> AudioChunk:
        """Read the next chunk from the WAV file, simulating real-time delay.

        If the file is exhausted, raises StopAsyncIteration.
        """
        if self._wav is None:
            raise RuntimeError("FileAudioSource not started — call start() first")

        if self._finished:
            raise StopAsyncIteration("WAV file exhausted")

        raw_data = self._wav.readframes(
            self._bytes_per_chunk // (self._wav.getnchannels() * self._wav.getsampwidth())
        )

        if not raw_data:
            self._finished = True
            raise StopAsyncIteration("WAV file exhausted")

        # Convert to mono if stereo
        data = self._to_mono(raw_data, self._wav.getnchannels(), self._wav.getsampwidth())

        # Resample to target sample rate if needed
        actual_rate = self._wav.getframerate()
        if actual_rate != self._config.sample_rate:
            data = self._resample(data, actual_rate, self._config.sample_rate)

        duration_ms = self._config.chunk_duration_ms
        self._elapsed_ms += duration_ms

        chunk = AudioChunk(
            data=data,
            sample_rate=self._config.sample_rate,
            channels=1,
            duration_ms=float(duration_ms),
        )

        # Simulate real-time playback by sleeping for the chunk duration
        await asyncio.sleep(self._chunk_duration_s)

        return chunk

    async def stop(self) -> None:
        """Close the WAV file."""
        if self._wav is not None:
            self._wav.close()
            self._wav = None

    # --- Private helpers ---

    @staticmethod
    def _to_mono(data: bytes, n_channels: int, sample_width: int) -> bytes:
        """Convert multi-channel PCM to mono by averaging channels."""
        if n_channels == 1:
            return data

        import struct

        fmt = "<h" if sample_width == 2 else "<b"
        frame_size = n_channels * sample_width
        n_frames = len(data) // frame_size

        mono_samples: list[int] = []
        for i in range(n_frames):
            frame_start = i * frame_size
            channel_sum = 0
            for ch in range(n_channels):
                offset = frame_start + ch * sample_width
                (sample,) = struct.unpack_from(fmt, data, offset)
                channel_sum += sample
            mono_samples.append(channel_sum // n_channels)

        out = io.BytesIO()
        for s in mono_samples:
            out.write(struct.pack("<h", max(-32768, min(32767, s))))
        return out.getvalue()

    @staticmethod
    def _resample(data: bytes, src_rate: int, dst_rate: int) -> bytes:
        """Simple linear interpolation resampling (adequate for testing)."""
        import struct

        n_samples = len(data) // 2
        samples = struct.unpack(f"<{n_samples}h", data)

        ratio = src_rate / dst_rate
        new_length = int(n_samples / ratio)

        resampled: list[int] = []
        for i in range(new_length):
            src_pos = i * ratio
            idx = int(src_pos)
            frac = src_pos - idx

            if idx + 1 < n_samples:
                val = samples[idx] * (1 - frac) + samples[idx + 1] * frac
            else:
                val = float(samples[min(idx, n_samples - 1)])

            resampled.append(max(-32768, min(32767, int(val))))

        return struct.pack(f"<{len(resampled)}h", *resampled)

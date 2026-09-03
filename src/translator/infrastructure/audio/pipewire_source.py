"""PipeWireAudioSource: Linux system audio capture via PipeWire/PulseAudio.

Uses subprocess calls to `pw-record` (PipeWire) or `parec` (PulseAudio)
to capture system audio in monitor mode. Falls back from PipeWire to
PulseAudio automatically.

Requirements:
  - System packages: pipewire (pw-record) or pulseaudio-utils (parec)
  - No extra pip packages required.
"""

from __future__ import annotations

import asyncio
import shutil
import struct
from typing import TYPE_CHECKING

from translator.core.events import AudioChunk

if TYPE_CHECKING:
    from translator.core.config import AudioConfig


class PipeWireAudioSource:
    """AudioSource implementation using PipeWire/PulseAudio on Linux.

    Launches a subprocess that captures the system monitor source and
    pipes raw PCM data to stdout, which we read asynchronously.

    This class satisfies the AudioSource Protocol (structural subtyping).
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._process: asyncio.subprocess.Process | None = None
        self._backend: str = ""  # "pipewire" or "pulseaudio"
        self._elapsed_ms: float = 0.0

    async def start(self) -> None:
        """Detect available backend and start the capture subprocess."""
        self._backend = self._detect_backend()
        cmd = self._build_command()

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._elapsed_ms = 0.0

    async def read_chunk(self) -> AudioChunk:
        """Read the next audio chunk from the capture subprocess."""
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("PipeWireAudioSource not started — call start() first")

        # Calculate bytes needed for one chunk
        bytes_per_sample = 2  # 16-bit
        samples_per_chunk = int(
            self._config.sample_rate * self._config.chunk_duration_ms / 1000
        )
        bytes_needed = samples_per_chunk * bytes_per_sample * self._config.channels

        # Read exactly the needed bytes
        data = await self._process.stdout.readexactly(bytes_needed)

        # Convert to mono if stereo
        if self._config.channels > 1:
            data = self._to_mono(data, self._config.channels)

        duration_ms = self._config.chunk_duration_ms
        self._elapsed_ms += duration_ms

        return AudioChunk(
            data=data,
            sample_rate=self._config.sample_rate,
            channels=1,
            duration_ms=float(duration_ms),
        )

    async def stop(self) -> None:
        """Terminate the capture subprocess."""
        if self._process is not None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except TimeoutError:
                self._process.kill()
            self._process = None

    # --- Private helpers ---

    def _detect_backend(self) -> str:
        """Detect whether PipeWire or PulseAudio is available."""
        if shutil.which("pw-record"):
            return "pipewire"
        if shutil.which("parec"):
            return "pulseaudio"
        raise RuntimeError(
            "Neither PipeWire (pw-record) nor PulseAudio (parec) found. "
            "Install one of: pipewire, pulseaudio-utils"
        )

    def _build_command(self) -> list[str]:
        """Build the subprocess command for audio capture."""
        rate = str(self._config.sample_rate)
        channels = str(self._config.channels)

        if self._backend == "pipewire":
            # pw-record captures from a monitor node
            # --target=0 captures the default audio sink's monitor
            cmd = [
                "pw-record",
                "--format", "s16",
                "--rate", rate,
                "--channels", channels,
                "--target", "0",
                "-",  # Output to stdout
            ]
        else:
            # parec captures from the PulseAudio monitor source
            cmd = [
                "parec",
                "--format=s16le",
                f"--rate={rate}",
                f"--channels={channels}",
                "--raw",
                "--latency-msec=30",
            ]

        return cmd

    @staticmethod
    def _to_mono(data: bytes, n_channels: int) -> bytes:
        """Convert multi-channel 16-bit PCM to mono by averaging."""
        if n_channels == 1:
            return data

        n_frames = len(data) // (n_channels * 2)
        mono_samples: list[int] = []
        for i in range(n_frames):
            total = 0
            for ch in range(n_channels):
                offset = (i * n_channels + ch) * 2
                (sample,) = struct.unpack_from("<h", data, offset)
                total += sample
            mono_samples.append(total // n_channels)

        return struct.pack(f"<{len(mono_samples)}h", *mono_samples)

"""WASAPIAudioSource: Windows system audio capture via WASAPI loopback.

Uses PyAudioWPatch to capture system audio (what you hear) without
requiring virtual audio cables. This is the primary audio source
for Windows.

Requirements:
  pip install PyAudioWPatch  (or install with [windows] extra)
"""

from __future__ import annotations

import asyncio
import struct
from typing import TYPE_CHECKING, Any

from translator.core.events import AudioChunk

if TYPE_CHECKING:
    from translator.core.config import AudioConfig


class WASAPIAudioSource:
    """AudioSource implementation using WASAPI loopback on Windows.

    Captures the system's default audio output device in loopback mode,
    producing 16kHz mono PCM AudioChunks.

    This class satisfies the AudioSource Protocol (structural subtyping).
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._pa: Any = None  # pyaudiowpatch.PyAudio
        self._stream: Any = None  # pyaudiowpatch.Stream
        self._device_info: dict[str, Any] | None = None
        self._buffer: bytes = b""
        self._elapsed_ms: float = 0.0

    async def start(self) -> None:
        """Open the WASAPI loopback stream."""
        try:
            import pyaudiowpatch as pyaudio  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "PyAudioWPatch is required for WASAPI loopback. "
                "Install with: pip install PyAudioWPatch"
            ) from e

        self._pa = pyaudio.PyAudio()

        # Find the WASAPI loopback device
        self._device_info = self._find_loopback_device(self._pa)
        if self._device_info is None:
            raise RuntimeError(
                "No WASAPI loopback device found. Ensure you have a default "
                "audio output device configured in Windows."
            )

        device_rate = int(self._device_info["defaultSampleRate"])
        device_channels = int(self._device_info["maxInputChannels"])

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=device_channels,
            rate=device_rate,
            input=True,
            input_device_index=int(self._device_info["index"]),
            frames_per_buffer=int(device_rate * self._config.chunk_duration_ms / 1000),
            stream_callback=None,  # We use blocking reads with async wrapping
        )
        self._stream.start_stream()
        self._elapsed_ms = 0.0

    async def read_chunk(self) -> AudioChunk:
        """Read the next audio chunk from the WASAPI loopback stream."""
        if self._stream is None or self._device_info is None:
            raise RuntimeError("WASAPIAudioSource not started — call start() first")

        device_rate = int(self._device_info["defaultSampleRate"])
        device_channels = int(self._device_info["maxInputChannels"])
        frames_to_read = int(device_rate * self._config.chunk_duration_ms / 1000)

        # Read from WASAPI in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        raw_data: bytes = await loop.run_in_executor(
            None,
            lambda: self._stream.read(frames_to_read, exception_on_overflow=False),
        )

        # Convert to mono if multi-channel
        mono_data = self._to_mono(raw_data, device_channels)

        # Resample to target sample rate if needed
        if device_rate != self._config.sample_rate:
            mono_data = self._resample(mono_data, device_rate, self._config.sample_rate)

        duration_ms = self._config.chunk_duration_ms
        self._elapsed_ms += duration_ms

        return AudioChunk(
            data=mono_data,
            sample_rate=self._config.sample_rate,
            channels=1,
            duration_ms=float(duration_ms),
        )

    async def stop(self) -> None:
        """Stop the WASAPI stream and release resources."""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    # --- Private helpers ---

    def _find_loopback_device(self, pa: Any) -> dict[str, Any] | None:
        """Find the default WASAPI loopback device."""
        try:
            import pyaudiowpatch as pyaudio  # type: ignore[import-untyped]
        except ImportError:
            return None

        # Get default WASAPI output device, then find its loopback counterpart
        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            return None

        # If a specific device is not requested, try to find the loopback for the default output
        target_name = self._config.device_name
        if not target_name:
            default_loopback_idx = wasapi_info.get("defaultLoopbackDevice", -1)
            if default_loopback_idx is not None and default_loopback_idx >= 0:
                dev = pa.get_device_info_by_index(default_loopback_idx)
                return dict(dev)
            
            # Fallback: get name of default output device and search for its loopback
            default_out_idx = wasapi_info.get("defaultOutputDevice", -1)
            if default_out_idx is not None and default_out_idx >= 0:
                out_dev = pa.get_device_info_by_index(default_out_idx)
                # The loopback device will usually have the exact same name plus "[Loopback]"
                # So we use the original name as the target to search for
                target_name = out_dev.get("name", "").replace(" [Loopback]", "")

        # Otherwise, search for a loopback device matching the name
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if dev.get("maxInputChannels", 0) > 0 and dev.get("isLoopbackDevice", "[Loopback]" in dev.get("name", "")):
                if target_name and target_name.lower() not in dev["name"].lower():
                    continue
                return dict(dev)

        return None

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

    @staticmethod
    def _resample(data: bytes, src_rate: int, dst_rate: int) -> bytes:
        """Simple linear interpolation resampling."""
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

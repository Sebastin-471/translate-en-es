"""Audio Device Manager: Cross-platform audio device enumeration and management.

Provides a unified interface for listing, selecting, and monitoring
audio capture devices across Windows (WASAPI) and Linux (PipeWire/PulseAudio).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from translator.core.config import AudioConfig


class AudioBackend(str, Enum):
    """Supported audio backends."""

    WASAPI = "wasapi"
    PIPEWIRE = "pipewire"
    FILE = "file"


class DeviceType(str, Enum):
    """Type of audio device."""

    INPUT = "input"          # Microphone, line-in
    OUTPUT = "output"        # Speakers, headphones
    LOOPBACK = "loopback"    # Monitor of output device (for system audio capture)


@dataclass(frozen=True, slots=True)
class AudioDeviceInfo:
    """Information about an audio device."""

    id: str                    # Backend-specific device identifier
    name: str                  # Human-readable name
    device_type: DeviceType    # Input, output, or loopback
    backend: AudioBackend      # Which backend this device belongs to
    is_default: bool = False   # Whether this is the system default
    sample_rates: list[int] | None = None  # Supported sample rates
    max_channels: int = 2      # Maximum number of channels


@runtime_checkable
class AudioDeviceManager(Protocol):
    """Interface for audio device enumeration and management.

    Implementations are backend-specific (WASAPI, PipeWire, etc.).
    """

    async def list_devices(self, device_type: DeviceType | None = None) -> list[AudioDeviceInfo]:
        """List available audio devices.

        Args:
            device_type: Filter by device type (None = all).

        Returns:
            List of AudioDeviceInfo objects.
        """
        ...

    async def get_default_device(self, device_type: DeviceType) -> AudioDeviceInfo | None:
        """Get the default device for a given type.

        Args:
            device_type: The type of device to get default for.

        Returns:
            AudioDeviceInfo if found, None otherwise.
        """
        ...

    async def watch_for_changes(self) -> None:
        """Watch for device changes (hot-plug, default changes).

        This should be called once to start monitoring. Changes can be
        observed via the on_devices_changed callback.
        """
        ...

    def set_change_callback(self, callback: "Callable[[list[AudioDeviceInfo]], None]") -> None:
        """Set callback for device list changes.

        Args:
            callback: Called with updated device list when changes detected.
        """
        ...


class DeviceChangeEvent:
    """Event emitted when audio devices change."""

    def __init__(self, devices: list[AudioDeviceInfo]) -> None:
        self.devices = devices
        self.timestamp = __import__("time").monotonic()
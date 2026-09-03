"""Audio Device Manager Factory."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from translator.core.audio_device_manager import AudioDeviceManager


def get_device_manager() -> AudioDeviceManager:
    """Get the appropriate AudioDeviceManager for the current platform."""
    if sys.platform == "win32":
        from translator.infrastructure.audio.wasapi_device_manager import WASAPIDeviceManager
        return WASAPIDeviceManager()
    else:
        from translator.infrastructure.audio.pipewire_device_manager import PipeWireDeviceManager
        return PipeWireDeviceManager()
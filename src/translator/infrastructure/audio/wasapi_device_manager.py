"""WASAPI Audio Device Manager for Windows."""

from __future__ import annotations

import asyncio
from typing import Callable

import structlog

from translator.core.audio_device_manager import (
    AudioBackend,
    AudioDeviceInfo,
    AudioDeviceManager,
    DeviceType,
)

logger = structlog.get_logger(__name__)


class WASAPIDeviceManager:
    """WASAPI implementation of AudioDeviceManager.

    Enumerates audio devices using PyAudioWPatch (WASAPI).
    Supports loopback devices for system audio capture.
    """

    def __init__(self) -> None:
        self._change_callback: Callable[[list[AudioDeviceInfo]], None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._cached_devices: list[AudioDeviceInfo] = []
        self._running = False

    async def list_devices(self, device_type: DeviceType | None = None) -> list[AudioDeviceInfo]:
        """List available WASAPI audio devices."""
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            logger.warning("pyaudiowpatch_not_available")
            return []

        devices = []
        try:
            pa = pyaudio.PyAudio()

            # Get default loopback device info
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_loopback_idx = wasapi_info.get("defaultLoopbackDevice", -1)

            for i in range(pa.get_device_count()):
                try:
                    dev = pa.get_device_info_by_index(i)
                except Exception:
                    continue

                if dev.get("hostApi") != wasapi_info["index"]:
                    continue

                # Determine device type
                is_input = dev.get("maxInputChannels", 0) > 0
                is_output = dev.get("maxOutputChannels", 0) > 0
                is_loopback = dev.get("isLoopbackDevice", False) or "[Loopback]" in dev.get("name", "")

                if is_loopback:
                    dtype = DeviceType.LOOPBACK
                elif is_input:
                    dtype = DeviceType.INPUT
                elif is_output:
                    dtype = DeviceType.OUTPUT
                else:
                    continue

                if device_type and dtype != device_type:
                    continue

                info = AudioDeviceInfo(
                    id=str(i),
                    name=dev.get("name", f"Device {i}"),
                    device_type=dtype,
                    backend=AudioBackend.WASAPI,
                    is_default=(i == default_loopback_idx and dtype == DeviceType.LOOPBACK),
                    sample_rates=[int(dev.get("defaultSampleRate", 48000))],
                    max_channels=max(
                        dev.get("maxInputChannels", 0),
                        dev.get("maxOutputChannels", 0),
                    ),
                )
                devices.append(info)

            pa.terminate()
            self._cached_devices = devices
            return devices

        except Exception as e:
            logger.exception("wasapi_list_devices_failed", error=str(e))
            return []

    async def get_default_device(self, device_type: DeviceType) -> AudioDeviceInfo | None:
        """Get the default WASAPI device for a given type."""
        devices = await self.list_devices(device_type)
        for dev in devices:
            if dev.is_default:
                return dev
        # Fallback: return first matching device
        return devices[0] if devices else None

    async def watch_for_changes(self) -> None:
        """Start monitoring for device changes (polling-based)."""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    def set_change_callback(self, callback: Callable[[list[AudioDeviceInfo]], None]) -> None:
        """Set callback for device list changes."""
        self._change_callback = callback

    def stop_watching(self) -> None:
        """Stop monitoring for device changes."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()

    async def _monitor_loop(self) -> None:
        """Poll for device changes."""
        while self._running:
            try:
                await asyncio.sleep(2.0)  # Poll every 2 seconds
                new_devices = await self.list_devices()
                if new_devices != self._cached_devices:
                    self._cached_devices = new_devices
                    if self._change_callback:
                        try:
                            self._change_callback(new_devices)
                        except Exception:
                            logger.exception("device_change_callback_error")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("device_monitor_error")
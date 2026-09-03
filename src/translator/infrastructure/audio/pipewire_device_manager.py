"""PipeWire Audio Device Manager for Linux."""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Callable

import structlog

from translator.core.audio_device_manager import (
    AudioBackend,
    AudioDeviceInfo,
    AudioDeviceManager,
    DeviceType,
)

logger = structlog.get_logger(__name__)


class PipeWireDeviceManager:
    """PipeWire/PulseAudio implementation of AudioDeviceManager.

    Uses `pactl` and `pw-dump` for device enumeration.
    """

    def __init__(self) -> None:
        self._change_callback: Callable[[list[AudioDeviceInfo]], None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._cached_devices: list[AudioDeviceInfo] = []
        self._running = False

    async def list_devices(self, device_type: DeviceType | None = None) -> list[AudioDeviceInfo]:
        """List available PipeWire/PulseAudio devices."""
        devices = []

        # Try pw-dump first (more detailed)
        pw_devices = await self._get_pipewire_devices()
        if pw_devices:
            devices.extend(pw_devices)
        else:
            # Fallback to pactl
            pactl_devices = await self._get_pactl_devices()
            devices.extend(pactl_devices)

        # Filter by type if requested
        if device_type:
            devices = [d for d in devices if d.device_type == device_type]

        self._cached_devices = devices
        return devices

    async def get_default_device(self, device_type: DeviceType) -> AudioDeviceInfo | None:
        """Get the default PipeWire device for a given type."""
        devices = await self.list_devices(device_type)
        for dev in devices:
            if dev.is_default:
                return dev
        return devices[0] if devices else None

    async def watch_for_changes(self) -> None:
        """Start monitoring for device changes using pactl subscribe."""
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

    # --- Private methods ---

    async def _get_pipewire_devices(self) -> list[AudioDeviceInfo]:
        """Get devices using pw-dump (PipeWire native)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "pw-dump",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()

            if proc.returncode != 0:
                return []

            data = json.loads(stdout.decode())
            devices = []

            for obj in data:
                if obj.get("type") != "PipeWire:Interface:Node":
                    continue

                props = obj.get("info", {}).get("props", {})
                media_class = props.get("media.class", "")

                # Determine device type
                if "Audio/Source" in media_class:
                    dtype = DeviceType.INPUT
                elif "Audio/Sink" in media_class:
                    dtype = DeviceType.OUTPUT
                elif "Audio/Duplex" in media_class:
                    dtype = DeviceType.LOOPBACK
                else:
                    continue

                device_id = str(obj.get("id", ""))
                name = props.get("node.name", props.get("device.name", f"Device {device_id}"))
                is_default = props.get("priority.driver", 0) > 1000 or props.get("priority.session", 0) > 1000

                info = AudioDeviceInfo(
                    id=device_id,
                    name=name,
                    device_type=dtype,
                    backend=AudioBackend.PIPEWIRE,
                    is_default=is_default,
                )
                devices.append(info)

            return devices

        except (FileNotFoundError, json.JSONDecodeError, Exception):
            return []

    async def _get_pactl_devices(self) -> list[AudioDeviceInfo]:
        """Get devices using pactl (PulseAudio compatibility)."""
        devices = []

        try:
            # Get sources (inputs)
            proc = await asyncio.create_subprocess_exec(
                "pactl", "-f", "json", "list", "sources",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                sources = json.loads(stdout.decode())
                for src in sources:
                    props = src.get("properties", {})
                    device_id = str(src.get("index", ""))
                    name = props.get("device.description", props.get("device.name", f"Source {device_id}"))
                    is_default = src.get("flags", {}).get("default", False)

                    info = AudioDeviceInfo(
                        id=device_id,
                        name=name,
                        device_type=DeviceType.INPUT,
                        backend=AudioBackend.PIPEWIRE,
                        is_default=is_default,
                    )
                    devices.append(info)

            # Get sinks (outputs)
            proc = await asyncio.create_subprocess_exec(
                "pactl", "-f", "json", "list", "sinks",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                sinks = json.loads(stdout.decode())
                for sink in sinks:
                    props = sink.get("properties", {})
                    device_id = str(sink.get("index", ""))
                    name = props.get("device.description", props.get("device.name", f"Sink {device_id}"))
                    is_default = sink.get("flags", {}).get("default", False)

                    info = AudioDeviceInfo(
                        id=device_id,
                        name=name,
                        device_type=DeviceType.OUTPUT,
                        backend=AudioBackend.PIPEWIRE,
                        is_default=is_default,
                    )
                    devices.append(info)

        except (FileNotFoundError, json.JSONDecodeError, Exception):
            pass

        return devices

    async def _monitor_loop(self) -> None:
        """Monitor for device changes using pactl subscribe."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "pactl", "subscribe",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            while self._running and proc.stdout:
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
                    if not line:
                        break

                    line = line.decode().strip()
                    if "source" in line or "sink" in line or "card" in line:
                        # Device changed, refresh list
                        new_devices = await self.list_devices()
                        if new_devices != self._cached_devices:
                            self._cached_devices = new_devices
                            if self._change_callback:
                                try:
                                    self._change_callback(new_devices)
                                except Exception:
                                    logger.exception("device_change_callback_error")

                except asyncio.TimeoutError:
                    continue
                except Exception:
                    logger.exception("pactl_monitor_error")
                    break

        except FileNotFoundError:
            # Fallback to polling if pactl subscribe not available
            await self._poll_loop()

    async def _poll_loop(self) -> None:
        """Polling fallback for device changes."""
        while self._running:
            try:
                await asyncio.sleep(3.0)
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
                logger.exception("device_poll_error")
"""Global hotkey management for the translator overlay.

Provides cross-platform global hotkey registration using `pynput`.
Supports toggle overlay, pause/resume pipeline, and quit.

The hotkey listener runs in a background thread and communicates
with the async pipeline via a thread-safe asyncio.Queue.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import structlog

from translator.core.config import HotkeyConfig

logger = structlog.get_logger(__name__)


class HotkeyManager:
    """Manages global hotkeys for the translator application.

    Parses hotkey strings (e.g., "ctrl+shift+t") from config and
    registers them as global hotkeys using pynput.

    Thread-safe: the pynput listener runs in its own thread, and
    callbacks are dispatched to the asyncio event loop via a
    thread-safe asyncio.Queue consumed by a dedicated async task.
    """

    def __init__(
        self,
        config: HotkeyConfig,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._config = config
        self._loop = loop or asyncio.get_running_loop()
        self._listener: Any = None
        self._callbacks: dict[str, Callable[[], Coroutine[Any, Any, None]]] = {}
        self._hotkeys: dict[str, str] = {}
        self._active = False
        self._queue: asyncio.Queue[Callable[[], Coroutine[Any, Any, None]]] = asyncio.Queue()
        self._dispatch_task: asyncio.Task[None] | None = None

    def register(self, action: str, callback: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Register a callback for a named action.

        Args:
            action: Action name (must match a config field:
                    "toggle_overlay", "toggle_pause", "quit").
            callback: Async callable to invoke when the hotkey is pressed.
        """
        hotkey_str = getattr(self._config, action, None)
        if hotkey_str is None:
            logger.warning("hotkey_action_unknown", action=action)
            return

        self._callbacks[action] = callback
        self._hotkeys[action] = hotkey_str
        logger.debug("hotkey_registered", action=action, hotkey=hotkey_str)

    def start(self) -> None:
        """Start listening for global hotkeys."""
        try:
            from pynput import keyboard  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "hotkeys_disabled",
                reason="pynput not installed. Install with: pip install pynput",
            )
            return

        # Build pynput hotkey mapping
        pynput_hotkeys: dict[str, Callable[[], None]] = {}
        for action, hotkey_str in self._hotkeys.items():
            pynput_str = self._to_pynput_format(hotkey_str)
            callback = self._callbacks[action]
            pynput_hotkeys[pynput_str] = self._make_threadsafe_callback(callback)

        if not pynput_hotkeys:
            logger.info("no_hotkeys_configured")
            return

        self._listener = keyboard.GlobalHotKeys(pynput_hotkeys)
        self._listener.start()
        self._active = True

        # Start the dispatch task on the event loop
        self._dispatch_task = self._loop.create_task(self._dispatch_loop())

        logger.info(
            "hotkeys_started",
            hotkeys={a: h for a, h in self._hotkeys.items()},
        )

    def stop(self) -> None:
        """Stop the hotkey listener and dispatch task."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

        # Signal the dispatch loop to stop
        self._active = False
        if self._dispatch_task:
            self._dispatch_task.cancel()

        logger.info("hotkeys_stopped")

    @property
    def is_active(self) -> bool:
        """Return True if the hotkey listener is running."""
        return self._active

    # --- Private helpers ---

    def _make_threadsafe_callback(self, callback: Callable[[], Coroutine[Any, Any, None]]) -> Callable[[], None]:
        """Wrap a callback so it runs safely from the pynput thread."""
        def _wrapper() -> None:
            # Put the coroutine into the queue (thread-safe)
            try:
                self._queue.put_nowait(callback)
            except asyncio.QueueFull:
                logger.warning("hotkey_queue_full", callback=callback.__name__)

        return _wrapper

    async def _dispatch_loop(self) -> None:
        """Consume callbacks from the queue and execute them on the event loop."""
        while self._active:
            try:
                callback = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                try:
                    await callback()
                except Exception:
                    logger.exception("hotkey_callback_error", callback=callback.__name__)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("hotkey_dispatch_error")

    @staticmethod
    def _to_pynput_format(hotkey: str) -> str:
        """Convert 'ctrl+shift+t' format to pynput's '<ctrl>+<shift>+t' format."""
        parts = hotkey.lower().split("+")
        converted: list[str] = []
        for part in parts:
            part = part.strip()
            if part in ("ctrl", "shift", "alt", "cmd"):
                converted.append(f"<{part}>")
            else:
                converted.append(part)
        return "+".join(converted)

"""System tray icon manager for the translator using pystray.

Runs the pystray icon in a background thread and communicates
with the async pipeline via a thread-safe asyncio.Queue.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

import pystray
import structlog
from PIL import Image, ImageDraw

logger = structlog.get_logger(__name__)


def _create_image() -> Image.Image:
    """Generate a dynamic icon image for the system tray."""
    # Create a simple 64x64 blue square with "ES" text
    image = Image.new("RGB", (64, 64), color=(0, 120, 215))
    draw = ImageDraw.Draw(image)

    # Draw simple text (Pillow default font is very basic, but works anywhere)
    draw.text((16, 24), "EN-ES", fill=(255, 255, 255))

    return image


class SystemTray:
    """Manages the system tray icon and background thread.

    Thread-safe: the pystray runs in its own thread, and
    callbacks are dispatched to the asyncio event loop via a
    thread-safe asyncio.Queue consumed by a dedicated async task.
    """

    def __init__(
        self,
        on_quit: Callable[[], Coroutine[Any, Any, None]],
        on_settings: Callable[[], Coroutine[Any, Any, None]],
        on_pause: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        self._on_quit = on_quit
        self._on_settings = on_settings
        self._on_pause = on_pause
        self._icon: pystray.Icon | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[Callable[[], Coroutine[Any, Any, None]]] = asyncio.Queue()
        self._dispatch_task: asyncio.Task[None] | None = None
        self._running = False

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start the system tray icon in a background thread."""
        self._loop = loop or asyncio.get_running_loop()
        self._running = True

        # Start the dispatch task on the event loop
        self._dispatch_task = self._loop.create_task(self._dispatch_loop())

        # Start pystray in a separate thread
        self._thread = threading.Thread(target=self._run_tray, daemon=True, name="system-tray")
        self._thread.start()

        logger.info("system_tray_started")

    def stop(self) -> None:
        """Stop the system tray icon and dispatch task."""
        self._running = False
        if self._icon is not None:
            self._icon.stop()
        if self._dispatch_task:
            self._dispatch_task.cancel()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("system_tray_stopped")

    # --- Private methods ---

    def _run_tray(self) -> None:
        """Run the pystray icon (blocking call in background thread)."""
        logger.info("system_tray_running")

        menu = pystray.Menu(
            pystray.MenuItem("Settings...", self._action_settings),
            pystray.MenuItem("Pause / Resume", self._action_pause),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._action_quit),
        )

        self._icon = pystray.Icon(
            "translate-en-es",
            _create_image(),
            "Live Translator",
            menu=menu
        )

        # This will block until the icon is stopped
        self._icon.run()

    def _make_threadsafe_callback(
        self, callback: Callable[[], Coroutine[Any, Any, None]]
    ) -> Callable[[], None]:
        """Wrap a callback so it runs safely from the pystray thread."""
        def _wrapper() -> None:
            # Put the coroutine into the queue (thread-safe)
            try:
                self._queue.put_nowait(callback)
            except asyncio.QueueFull:
                logger.warning("tray_queue_full", callback=callback.__name__)

        return _wrapper

    async def _dispatch_loop(self) -> None:
        """Consume callbacks from the queue and execute them on the event loop."""
        while self._running:
            try:
                callback = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                try:
                    await callback()
                except Exception:
                    logger.exception("tray_callback_error", callback=callback.__name__)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("tray_dispatch_error")

    # --- Callbacks (called from pystray thread) ---

    def _action_quit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        logger.info("tray_action_quit")
        self._make_threadsafe_callback(self._on_quit)()

    def _action_settings(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        logger.info("tray_action_settings")
        self._make_threadsafe_callback(self._on_settings)()

    def _action_pause(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        logger.info("tray_action_pause")
        self._make_threadsafe_callback(self._on_pause)()

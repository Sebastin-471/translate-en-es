"""System tray icon manager for the translator using pystray."""

from __future__ import annotations

import asyncio
from typing import Callable

import pystray
import structlog
from PIL import Image, ImageDraw, ImageFont

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
    """Manages the system tray icon and background thread."""

    def __init__(
        self,
        on_quit: Callable[[], None],
        on_settings: Callable[[], None],
        on_pause: Callable[[], None],
    ) -> None:
        self._on_quit = on_quit
        self._on_settings = on_settings
        self._on_pause = on_pause
        self._icon: pystray.Icon | None = None

    def start(self) -> None:
        """Start the system tray icon (runs in current thread, blocks)."""
        logger.info("system_tray_starting")
        
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

    def stop(self) -> None:
        """Stop the system tray icon."""
        if self._icon is not None:
            self._icon.stop()
            logger.info("system_tray_stopped")

    # --- Callbacks ---

    def _action_quit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        logger.info("tray_action_quit")
        self._on_quit()
        # The icon will be stopped via the shutdown flow.

    def _action_settings(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        logger.info("tray_action_settings")
        self._on_settings()

    def _action_pause(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        logger.info("tray_action_pause")
        self._on_pause()

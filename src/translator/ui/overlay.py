"""TkinterOverlayRenderer: Always-on-top translucent subtitle overlay.

Renders translated subtitles as an always-on-top, borderless, translucent
overlay window using Tkinter. On Windows, uses `-transparentcolor` for
true transparency with opaque text. On Linux, uses `-alpha` for the
entire window.

The overlay runs Tkinter's mainloop in a separate thread (Tkinter
requires exclusive access to its thread). Communication with the async
pipeline happens via a thread-safe queue and `root.after()` polling.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from translator.core.events import TranslationResult

if TYPE_CHECKING:
    from translator.core.config import UIConfig

logger = structlog.get_logger(__name__)

# Platform-specific transparent color (Windows only)
_TRANSPARENT_COLOR = "#010101"


@dataclass
class _SubtitleLine:
    """A single subtitle line with expiry tracking."""

    original: str
    translated: str
    created_at: float
    sequence_id: str
    is_partial: bool
    label_translated: tk.Label | None = None
    label_original: tk.Label | None = None


class TkinterOverlayRenderer:
    """UIRenderer implementation using Tkinter for always-on-top subtitles.

    Displays a translucent overlay at the bottom (or top/center) of the
    screen with translated text and optionally the original text above it.

    This class satisfies the UIRenderer Protocol (structural subtyping).

    Threading model:
      - The async pipeline calls `show()` / `clear()` from the asyncio
        event loop thread.
      - These methods put commands into a thread-safe queue.
      - The Tkinter thread polls this queue via `root.after()` and
        updates the UI.
    """

    def __init__(self, config: UIConfig) -> None:
        self._config = config
        self._command_queue: queue.Queue[tuple[str, TranslationResult | None]] = queue.Queue()
        self._root: tk.Tk | None = None
        self._frame: tk.Frame | None = None
        self._lines: list[_SubtitleLine] = []
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._stopped = threading.Event()

    async def show(self, translation: TranslationResult) -> None:
        """Queue a translation for display in the overlay."""
        self._command_queue.put(("show", translation))

    async def clear(self) -> None:
        """Queue a clear command for the overlay."""
        self._command_queue.put(("clear", None))

    async def start(self) -> None:
        """Launch the Tkinter overlay in a background thread."""
        if not self._config.enabled:
            logger.info("overlay_disabled")
            return

        self._thread = threading.Thread(
            target=self._run_tk,
            daemon=True,
            name="overlay-ui",
        )
        self._thread.start()

        # Wait for Tkinter to initialize
        self._started.wait(timeout=5.0)
        if not self._started.is_set():
            logger.warning("overlay_start_timeout")
        else:
            logger.info("overlay_started", position=self._config.position)
            # Inject a startup message for visual feedback
            welcome_msg = TranslationResult(
                original_text="[System: Translator started. Listening for English audio...]",
                translated_text="[Sistema: Traductor iniciado. Escuchando audio en inglés...]",
                source_language="en",
                target_language="es",
                segment_start_ms=0.0,
                segment_end_ms=0.0,
                processing_time_ms=0.0,
                sequence_id="system-startup",
            )
            await self.show(welcome_msg)

    async def stop(self) -> None:
        """Signal the Tkinter thread to quit."""
        self._command_queue.put(("quit", None))
        self._stopped.wait(timeout=5.0)

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        logger.info("overlay_stopped")

    # --- Tkinter thread ---

    def _run_tk(self) -> None:
        """Tkinter main loop running in a background thread."""
        try:
            self._root = tk.Tk()
            root = self._root

            root.title("Translator Overlay")
            root.overrideredirect(True)
            root.attributes("-topmost", True)

            # Platform-specific transparency
            if sys.platform == "win32":
                root.configure(bg=_TRANSPARENT_COLOR)
                root.attributes("-transparentcolor", _TRANSPARENT_COLOR)
            else:
                root.attributes("-alpha", self._config.background_opacity)
                root.configure(bg="black")

            # Position the window
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            win_w = screen_w - 2 * self._config.margin_x
            win_h = (self._config.font_size * 3) * self._config.max_lines

            x = self._config.margin_x
            if self._config.position == "bottom":
                y = screen_h - win_h - self._config.margin_y
            elif self._config.position == "top":
                y = self._config.margin_y
            else:  # center
                y = (screen_h - win_h) // 2

            root.geometry(f"{win_w}x{win_h}+{x}+{y}")

            # Container frame
            if sys.platform == "win32":
                self._frame = tk.Frame(root, bg=_TRANSPARENT_COLOR)
            else:
                self._frame = tk.Frame(root, bg="black")
            self._frame.pack(fill=tk.BOTH, expand=True, anchor=tk.S)

            # Drag-to-move support
            self._drag_start_x = 0
            self._drag_start_y = 0
            root.bind("<Button-1>", self._on_drag_start)
            root.bind("<B1-Motion>", self._on_drag_motion)
            self._frame.bind("<Button-1>", self._on_drag_start)
            self._frame.bind("<B1-Motion>", self._on_drag_motion)

            # Start polling the command queue
            self._poll_commands()

            self._started.set()
            root.mainloop()

        except Exception:
            logger.exception("overlay_error")
        finally:
            self._stopped.set()

    def _poll_commands(self) -> None:
        """Poll the command queue from the Tkinter thread."""
        if self._root is None:
            return

        try:
            while True:
                cmd, data = self._command_queue.get_nowait()
                if cmd == "show" and data is not None:
                    self._add_subtitle(data)
                elif cmd == "clear":
                    self._clear_subtitles()
                elif cmd == "open_settings":
                    self._show_settings()
                elif cmd == "quit":
                    self._root.quit()
                    return
        except queue.Empty:
            pass

        # Remove expired lines
        self._expire_old_lines()

        # Schedule next poll (50ms ≈ 20fps)
        self._root.after(50, self._poll_commands)

    def _truncate_text(self, text: str, max_words: int = 25) -> str:
        """Truncate text to keep only the last max_words to prevent UI overflow."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return "... " + " ".join(words[-max_words:])

    def _add_subtitle(self, translation: TranslationResult) -> None:
        """Add a new subtitle line to the overlay."""
        if self._frame is None or self._root is None:
            return

        # Check if we already have a line for this sequence_id
        existing_line = next(
            (line for line in self._lines if line.sequence_id == translation.sequence_id), None
        )

        display_translated = self._truncate_text(translation.translated_text)
        display_original = self._truncate_text(translation.original_text)

        if existing_line:
            # Update existing line
            existing_line.original = translation.original_text
            existing_line.translated = translation.translated_text
            existing_line.created_at = time.monotonic()
            existing_line.is_partial = translation.is_partial

            if existing_line.label_translated:
                existing_line.label_translated.config(text=display_translated)

            if existing_line.label_original and self._config.show_original:
                existing_line.label_original.config(text=display_original)

            return

        bg_color = self._config.background_color
        if sys.platform == "win32":
            # On Windows, use the actual background color for the text band
            bg_color = self._config.background_color

        line = _SubtitleLine(
            original=translation.original_text,
            translated=translation.translated_text,
            created_at=time.monotonic(),
            sequence_id=translation.sequence_id,
            is_partial=translation.is_partial,
        )

        # Create translated text label
        line.label_translated = tk.Label(
            self._frame,
            text=display_translated,
            fg=self._config.text_color,
            bg=bg_color,
            font=(self._config.font_family, self._config.font_size, "bold"),
            wraplength=self._root.winfo_width() - 40,
            justify=tk.CENTER,
            padx=10,
            pady=2,
        )
        line.label_translated.pack(side=tk.BOTTOM, fill=tk.X)

        # Optionally show original text
        if self._config.show_original:
            smaller_size = max(10, self._config.font_size - 4)
            line.label_original = tk.Label(
                self._frame,
                text=display_original,
                fg="#AAAAAA",
                bg=bg_color,
                font=(self._config.font_family, smaller_size, "italic"),
                wraplength=self._root.winfo_width() - 40,
                justify=tk.CENTER,
                padx=10,
                pady=1,
            )
            line.label_original.pack(side=tk.BOTTOM, fill=tk.X)

        self._lines.append(line)

        # Enforce max lines
        while len(self._lines) > self._config.max_lines:
            self._remove_line(self._lines[0])
            self._lines.pop(0)

    def _expire_old_lines(self) -> None:
        """Remove subtitle lines that have exceeded the fade timeout."""
        now = time.monotonic()
        expired = [
            line
            for line in self._lines
            if now - line.created_at > self._config.fade_after_seconds
        ]
        for line in expired:
            self._remove_line(line)
            self._lines.remove(line)

    def _remove_line(self, line: _SubtitleLine) -> None:
        """Remove a subtitle line's widgets from the frame."""
        if line.label_translated is not None:
            line.label_translated.destroy()
        if line.label_original is not None:
            line.label_original.destroy()

    def _clear_subtitles(self) -> None:
        """Remove all subtitle lines."""
        for line in self._lines:
            self._remove_line(line)
        self._lines.clear()

    async def open_settings(self) -> None:
        """Queue a command to open the settings window."""
        self._command_queue.put(("open_settings", None))

    def _show_settings(self) -> None:
        """Open the CustomTkinter settings window."""
        if self._root is None:
            return

        from translator.ui.settings import SettingsWindow

        def on_save(new_config):
            # In a real app we'd save to disk and restart pipeline.
            # For now, we just log it.
            logger.info("settings_saved", new_config=new_config)

        SettingsWindow(self._root, self._config, on_save)

    # --- Drag-to-move handlers ---

    def _on_drag_start(self, event: tk.Event) -> None:
        """Record the starting position of a drag."""
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag_motion(self, event: tk.Event) -> None:
        """Move the overlay window as the user drags."""
        if self._root is None:
            return
        x = self._root.winfo_x() + (event.x - self._drag_start_x)
        y = self._root.winfo_y() + (event.y - self._drag_start_y)
        self._root.geometry(f"+{x}+{y}")

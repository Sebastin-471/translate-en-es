"""Settings Window for the translator using CustomTkinter."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

import customtkinter as ctk

from translator.core.config import AppConfig


class SettingsWindow(ctk.CTkToplevel):
    """Settings dialog for the application."""

    def __init__(self, parent: tk.Tk, config: AppConfig, on_save: Callable[[AppConfig], None] | None = None) -> None:
        super().__init__(parent)
        self.title("Translator Settings")
        self.geometry("500x600")

        # Make the window modal and always on top
        self.transient(parent)
        self.grab_set()
        self.attributes("-topmost", True)

        self._config = config
        self._on_save = on_save

        self._build_ui()

    def _build_ui(self) -> None:
        # Define layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main scrollable frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.scrollable_frame.grid_columnconfigure(1, weight=1)

        # --- Audio Settings ---
        self._add_section_header("Audio Source", 0)

        ctk.CTkLabel(self.scrollable_frame, text="Device:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.device_var = ctk.StringVar(value="Default System Audio (Loopback)")
        self.device_menu = ctk.CTkOptionMenu(
            self.scrollable_frame,
            variable=self.device_var,
            values=["Default System Audio (Loopback)", "Microphone (Coming Soon)"]
        )
        self.device_menu.grid(row=1, column=1, sticky="ew", pady=(10, 0), padx=(10, 0))

        # --- AI Settings ---
        self._add_section_header("Artificial Intelligence", 2)

        ctk.CTkLabel(self.scrollable_frame, text="ASR Model (Whisper):").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.asr_var = ctk.StringVar(value=self._config.asr.model_size)
        self.asr_menu = ctk.CTkOptionMenu(
            self.scrollable_frame,
            variable=self.asr_var,
            values=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
        )
        self.asr_menu.grid(row=3, column=1, sticky="ew", pady=(10, 0), padx=(10, 0))

        ctk.CTkLabel(self.scrollable_frame, text="Compute Device:").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.device_type_var = ctk.StringVar(value=self._config.asr.device)
        self.device_type_menu = ctk.CTkOptionMenu(
            self.scrollable_frame,
            variable=self.device_type_var,
            values=["auto", "cuda", "cpu"]
        )
        self.device_type_menu.grid(row=4, column=1, sticky="ew", pady=(10, 0), padx=(10, 0))

        # --- Subtitles / Overlay Settings ---
        self._add_section_header("Overlay Appearance", 5)

        ctk.CTkLabel(self.scrollable_frame, text="Font Size:").grid(row=6, column=0, sticky="w", pady=(10, 0))
        self.font_size_var = ctk.IntVar(value=self._config.ui.font_size)
        self.font_slider = ctk.CTkSlider(
            self.scrollable_frame,
            from_=12,
            to=48,
            number_of_steps=36,
            variable=self.font_size_var
        )
        self.font_slider.grid(row=6, column=1, sticky="ew", pady=(10, 0), padx=(10, 0))

        self.show_original_var = ctk.BooleanVar(value=self._config.ui.show_original)
        self.show_original_switch = ctk.CTkSwitch(
            self.scrollable_frame,
            text="Show Original Text (English)",
            variable=self.show_original_var
        )
        self.show_original_switch.grid(row=7, column=0, columnspan=2, sticky="w", pady=(15, 0))

        # --- Buttons ---
        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=20)
        self.button_frame.grid_columnconfigure((0, 1), weight=1)

        self.cancel_btn = ctk.CTkButton(self.button_frame, text="Cancel", fg_color="gray", hover_color="darkgray", command=self.destroy)
        self.cancel_btn.grid(row=0, column=0, padx=10)

        self.save_btn = ctk.CTkButton(self.button_frame, text="Save & Restart", command=self._on_save_clicked)
        self.save_btn.grid(row=0, column=1, padx=10)

    def _add_section_header(self, text: str, row: int) -> None:
        header = ctk.CTkLabel(
            self.scrollable_frame,
            text=text,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        header.grid(row=row, column=0, columnspan=2, sticky="w", pady=(20, 5))

    def _on_save_clicked(self) -> None:
        # Update config object
        self._config.asr.model_size = self.asr_var.get()
        self._config.asr.device = self.device_type_var.get()
        self._config.ui.font_size = self.font_size_var.get()
        self._config.ui.show_original = self.show_original_var.get()

        # Execute callback if provided
        if self._on_save:
            self._on_save(self._config)

        self.destroy()

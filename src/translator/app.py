"""Composition root: assembles the entire translation pipeline.

This is the ONLY module in the project that imports from both core/
and infrastructure/. It reads config, detects the platform, instantiates
concrete engines, and wires them into the PipelineOrchestrator.

Import graph:
  app.py → core/* (interfaces, config, events, plugins)
  app.py → infrastructure/* (concrete implementations)
  app.py → pipeline/* (orchestrator)
  app.py → ui/* (overlay, hotkeys)
  app.py → config/* (loader)
  app.py → logging/* (setup)

No other module in the project imports from infrastructure/ directly.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from translator.config.loader import ConfigWatcher, load_config
from translator.core.config import AppConfig
from translator.core.plugins import EngineRegistry, register_builtin_engines
from translator.logging.setup import setup_logging
from translator.pipeline.orchestrator import PipelineOrchestrator

if TYPE_CHECKING:
    from translator.core.interfaces import (
        ASREngine,
        AudioSource,
        MTEngine,
        UIRenderer,
        VADEngine,
    )
    from translator.core.model_manager import ModelManager

logger = structlog.get_logger(__name__)


# Register built-in engines
register_builtin_engines()


def _create_audio_source(config: AppConfig) -> AudioSource:
    """Create the appropriate AudioSource based on config and platform."""
    backend = config.audio.backend

    # Handle mock mode
    if config.pipeline.mock_mode:
        return EngineRegistry.get_audio_source("mock")(config.audio)

    # Handle file backend
    if backend == "file":
        return EngineRegistry.get_audio_source("file")(config.audio)

    # Platform-specific backends
    if backend == "wasapi":
        if sys.platform != "win32":
            raise RuntimeError(
                f"WASAPI backend requires Windows, but running on {sys.platform}. "
                f"Set audio.backend to 'pipewire' for Linux."
            )
        return EngineRegistry.get_audio_source("wasapi")(config.audio)

    if backend == "pipewire":
        if sys.platform == "win32":
            raise RuntimeError(
                "PipeWire backend requires Linux. "
                "Set audio.backend to 'wasapi' for Windows."
            )
        return EngineRegistry.get_audio_source("pipewire")(config.audio)

    raise ValueError(f"Unknown audio backend: '{backend}'")


def _create_vad_engine(config: AppConfig, model_manager: ModelManager) -> VADEngine:
    """Create the VAD engine."""
    if config.pipeline.mock_mode:
        return EngineRegistry.get_vad_engine("mock")(config.vad, model_manager)
    return EngineRegistry.get_vad_engine("silero")(config.vad, model_manager)


def _create_asr_engine(config: AppConfig, model_manager: ModelManager) -> ASREngine:
    """Create the ASR engine."""
    if config.pipeline.mock_mode:
        return EngineRegistry.get_asr_engine("mock")(config.asr, model_manager)
    return EngineRegistry.get_asr_engine("whisper")(config.asr, model_manager)


def _create_mt_engine(config: AppConfig, model_manager: ModelManager) -> MTEngine:
    """Create the MT engine."""
    if config.pipeline.mock_mode:
        return EngineRegistry.get_mt_engine("mock")(config.mt, model_manager)
    return EngineRegistry.get_mt_engine("marian")(config.mt, model_manager)


def _create_ui_renderer(config: AppConfig) -> UIRenderer:
    """Create the UI renderer."""
    if config.pipeline.mock_mode or not config.ui.enabled:
        return EngineRegistry.get_ui_renderer("mock")(config.ui)
    return EngineRegistry.get_ui_renderer("tkinter")(config.ui)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="translate-en-es",
        description="Real-time system audio translation (EN→ES) for video calls",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="Path to base YAML configuration file (default: config/base.yaml)",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["development", "production", "testing"],
        help="Environment name (overrides TRANSLATOR_ENV env var)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (simulated engines, no GPU required)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level from config",
    )
    parser.add_argument(
        "--no-hot-reload",
        action="store_true",
        help="Disable configuration hot-reload",
    )
    return parser.parse_args()


class Application:
    """Main application class managing lifecycle and config hot-reload."""

    def __init__(self, config: AppConfig, args: argparse.Namespace) -> None:
        self._config = config
        self._args = args
        self._pipeline: PipelineOrchestrator | None = None
        self._model_manager: ModelManager | None = None
        self._hotkeys = None
        self._tray = None
        self._config_watcher: ConfigWatcher | None = None
        self._device_manager = None
        self._running = False

    async def start(self) -> None:
        """Start the application and pipeline."""
        self._running = True

        logger.info(
            "app_starting",
            platform=platform.system(),
            python=platform.python_version(),
            mock_mode=self._config.pipeline.mock_mode,
            asr_model=self._config.asr.model_size,
            mt_source=self._config.mt.source_language,
            mt_target=self._config.mt.target_language,
        )

        # --- Initialize GPU Model Manager ---
        from translator.infrastructure.gpu.gpu_model_manager import GPUModelManager

        self._model_manager = GPUModelManager(max_vram_mb=self._config.gpu.max_vram_mb)

        # --- Initialize Audio Device Manager ---
        from translator.infrastructure.audio.device_manager_factory import get_device_manager

        self._device_manager = get_device_manager()
        await self._device_manager.watch_for_changes()
        self._device_manager.set_change_callback(self._on_devices_changed)
        logger.info("audio_device_manager_started")

        # --- Dependency Injection: Create concrete implementations ---
        await self._create_engines()

        # --- Assemble Pipeline (Orchestrator receives only interfaces) ---
        self._pipeline = PipelineOrchestrator(
            audio_source=self._audio_source,
            vad_engine=self._vad_engine,
            asr_engine=self._asr_engine,
            mt_engine=self._mt_engine,
            ui_renderer=self._ui_renderer,
            config=self._config,
        )

        # --- Setup hotkeys ---
        loop = asyncio.get_running_loop()
        from translator.ui.hotkeys import HotkeyManager

        self._hotkeys = HotkeyManager(self._config.hotkeys, loop=loop)
        self._hotkeys.register("toggle_pause", self._on_toggle_pause)
        self._hotkeys.register("quit", self._on_quit)
        self._hotkeys.start()

        # --- Setup system tray ---
        from translator.ui.tray import SystemTray

        self._tray = SystemTray(
            on_quit=self._on_quit,
            on_settings=self._on_settings,
            on_pause=self._on_toggle_pause,
        )
        loop = asyncio.get_running_loop()
        self._tray.start(loop=loop)

        # --- Setup signal handlers for graceful shutdown ---
        def _signal_handler() -> None:
            logger.info("signal_received", signal="SIGINT/SIGTERM")
            asyncio.ensure_future(self.stop())

        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _signal_handler)

        # --- Setup config hot-reload ---
        if not self._args.no_hot_reload and self._config.config.watch_for_changes:
            await self._setup_config_watcher()

        # --- Start and run pipeline ---
        try:
            await self._pipeline.start()
            await self._pipeline.run()
        except KeyboardInterrupt:
            logger.info("keyboard_interrupt")
        finally:
            await self.stop()

    async def _create_engines(self) -> None:
        """Create or recreate all engines with current config."""
        self._audio_source = _create_audio_source(self._config)
        self._vad_engine = _create_vad_engine(self._config, self._model_manager)
        self._asr_engine = _create_asr_engine(self._config, self._model_manager)
        self._mt_engine = _create_mt_engine(self._config, self._model_manager)
        self._ui_renderer = _create_ui_renderer(self._config)

        logger.info(
            "engines_created",
            audio=type(self._audio_source).__name__,
            vad=type(self._vad_engine).__name__,
            asr=type(self._asr_engine).__name__,
            mt=type(self._mt_engine).__name__,
            ui=type(self._ui_renderer).__name__,
        )

    async def _setup_config_watcher(self) -> None:
        """Set up configuration file watching for hot-reload."""
        config_path = Path(self._args.config)
        watch_paths = [config_path]
        base_path = config_path.parent / "base.yaml"
        if base_path != config_path:
            watch_paths.append(base_path)
        env_path = config_path.parent / f"{self._get_env()}.yaml"
        if env_path.exists():
            watch_paths.append(env_path)

        def on_config_change(event):
            """Handle config change event."""
            asyncio.create_task(self._handle_config_change(event))

        self._config_watcher = ConfigWatcher(watch_paths, on_config_change)
        await self._config_watcher.start(self._config)

    def _get_env(self) -> str:
        """Get current environment name."""
        if self._args.env:
            return self._args.env
        return os.environ.get("TRANSLATOR_ENV", "development")

    async def _handle_config_change(self, event) -> None:
        """Handle configuration change by updating affected components."""
        logger.info("config_change_detected", changed_sections=list(event.changed_sections))

        # Update logging if changed
        if "logging" in event.changed_sections:
            setup_logging(self._config.logging)
            logger.info("logging_reconfigured")

        # Update UI renderer if UI config changed
        if "ui" in event.changed_sections and self._pipeline:
            # UI renderer changes require pipeline restart for now
            logger.info("ui_config_changed_requires_restart")
            # Could implement hot-swap for UI renderer in future

        # Update VAD engine if VAD config changed
        if "vad" in event.changed_sections and hasattr(self._vad_engine, "update_config"):
            await self._vad_engine.update_config(event.new_config.vad)
            logger.info("vad_engine_reconfigured")

        # Update pipeline queue size if pipeline config changed
        if "pipeline" in event.changed_sections and self._pipeline:
            logger.info("pipeline_config_changed_requires_restart")

        # Log other changes
        for section in event.changed_sections:
            if section not in ("logging", "ui", "vad", "pipeline"):
                logger.info("config_section_changed", section=section)

    def _on_devices_changed(self, devices) -> None:
        """Handle audio device list changes."""
        logger.info("audio_devices_changed", count=len(devices))
        # Log available loopback devices
        loopback_devices = [d for d in devices if d.device_type.value == "loopback"]
        if loopback_devices:
            logger.info("available_loopback_devices", devices=[d.name for d in loopback_devices])
        # Could implement auto-fallback logic here in the future

    async def stop(self) -> None:
        """Gracefully stop the application."""
        if not self._running:
            return

        self._running = False
        logger.info("app_stopping")

        # Stop config watcher
        if self._config_watcher:
            await self._config_watcher.stop()

        # Stop device manager
        if self._device_manager:
            self._device_manager.stop_watching()

        # Stop pipeline
        if self._pipeline:
            await self._pipeline.stop()

        # Stop UI
        if self._tray:
            self._tray.stop()
        if self._hotkeys:
            self._hotkeys.stop()

        logger.info("app_stopped")

    async def _on_toggle_pause(self) -> None:
        """Handle pause toggle from hotkey/tray."""
        if self._pipeline:
            await self._pipeline.toggle_pause()

    async def _on_quit(self) -> None:
        """Handle quit from hotkey/tray."""
        await self.stop()

    async def _on_settings(self) -> None:
        """Handle settings from tray."""
        if self._ui_renderer and hasattr(self._ui_renderer, "open_settings"):
            await self._ui_renderer.open_settings()


async def _async_main(config: AppConfig, args: argparse.Namespace) -> None:
    """Async entry point: assemble and run the application."""
    app = Application(config, args)
    await app.start()


def main() -> None:
    """CLI entry point: parse args, load config, run pipeline."""
    args = _parse_args()

    # Load configuration
    try:
        config = load_config(args.config, env=args.env)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        print("Create one from config/base.yaml or use --mock", file=sys.stderr)
        # Fall back to defaults if --mock is used
        if args.mock:
            config = AppConfig()
        else:
            sys.exit(1)

    # Apply CLI overrides
    if args.mock:
        config.pipeline.mock_mode = True
    if args.log_level:
        config.logging.level = args.log_level

    # Setup logging (must be done before any logging calls)
    setup_logging(config.logging)

    # Run the async application
    try:
        asyncio.run(_async_main(config, args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

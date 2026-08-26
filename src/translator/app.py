"""Composition root: assembles the entire translation pipeline.

This is the ONLY module in the project that imports from both core/
and infrastructure/. It reads config, detects the platform, instantiates
concrete engines, and wires them into the PipelineOrchestrator.

Import graph:
  app.py → core/* (interfaces, config, events)
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
import platform
import signal
import sys
from typing import TYPE_CHECKING

import structlog

from translator.config.loader import load_config
from translator.core.config import AppConfig
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


def _create_audio_source(config: AppConfig) -> AudioSource:
    """Create the appropriate AudioSource based on config and platform."""
    backend = config.audio.backend

    if backend == "file":
        from translator.infrastructure.audio.file_source import FileAudioSource

        return FileAudioSource(config.audio)

    if config.pipeline.mock_mode:
        from translator.infrastructure.mock.mock_audio import MockAudioSource

        return MockAudioSource(config.audio, generate_tone=True)

    if backend == "wasapi":
        if sys.platform != "win32":
            raise RuntimeError(
                f"WASAPI backend requires Windows, but running on {sys.platform}. "
                f"Set audio.backend to 'pipewire' for Linux."
            )
        from translator.infrastructure.audio.wasapi_source import WASAPIAudioSource

        return WASAPIAudioSource(config.audio)

    if backend == "pipewire":
        if sys.platform == "win32":
            raise RuntimeError(
                "PipeWire backend requires Linux. "
                "Set audio.backend to 'wasapi' for Windows."
            )
        from translator.infrastructure.audio.pipewire_source import PipeWireAudioSource

        return PipeWireAudioSource(config.audio)

    raise ValueError(f"Unknown audio backend: '{backend}'")


def _create_vad_engine(config: AppConfig, model_manager: ModelManager) -> VADEngine:
    """Create the VAD engine (real or mock)."""
    if config.pipeline.mock_mode:
        from translator.infrastructure.mock.mock_vad import MockVADEngine

        return MockVADEngine(
            delay_ms=config.pipeline.mock_delay_ms / 10,  # VAD is fast
        )

    from translator.infrastructure.vad.silero_vad import SileroVADEngine

    return SileroVADEngine(config.vad, model_manager, sample_rate=config.audio.sample_rate)


def _create_asr_engine(config: AppConfig, model_manager: ModelManager) -> ASREngine:
    """Create the ASR engine (real or mock)."""
    if config.pipeline.mock_mode:
        from translator.infrastructure.mock.mock_asr import MockASREngine

        return MockASREngine(
            delay_ms=config.pipeline.mock_delay_ms,
            language=config.asr.language,
        )

    from translator.infrastructure.asr.whisper_asr import WhisperASREngine

    return WhisperASREngine(config.asr, model_manager)


def _create_mt_engine(config: AppConfig, model_manager: ModelManager) -> MTEngine:
    """Create the MT engine (real or mock)."""
    if config.pipeline.mock_mode:
        from translator.infrastructure.mock.mock_mt import MockMTEngine

        return MockMTEngine(
            delay_ms=config.pipeline.mock_delay_ms / 2,
            source_language=config.mt.source_language,
            target_language=config.mt.target_language,
        )

    from translator.infrastructure.mt.marian_mt import MarianMTEngine

    return MarianMTEngine(config.mt, model_manager)


def _create_ui_renderer(config: AppConfig) -> UIRenderer:
    """Create the UI renderer (real overlay or mock logger)."""
    if config.pipeline.mock_mode or not config.ui.enabled:
        from translator.infrastructure.mock.mock_ui import MockUIRenderer

        return MockUIRenderer(show_original=config.ui.show_original)

    from translator.ui.overlay import TkinterOverlayRenderer

    return TkinterOverlayRenderer(config.ui)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="translate-en-es",
        description="Real-time system audio translation (EN→ES) for video calls",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
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
    return parser.parse_args()


async def _async_main(config: AppConfig) -> None:
    """Async entry point: assemble and run the pipeline."""
    logger.info(
        "app_starting",
        platform=platform.system(),
        python=platform.python_version(),
        mock_mode=config.pipeline.mock_mode,
        asr_model=config.asr.model_size,
        mt_source=config.mt.source_language,
        mt_target=config.mt.target_language,
    )

    # --- Initialize GPU Model Manager ---
    from translator.infrastructure.gpu.gpu_model_manager import GPUModelManager
    
    model_manager = GPUModelManager(max_vram_mb=config.gpu.max_vram_mb)

    # --- Dependency Injection: Create concrete implementations ---
    audio_source = _create_audio_source(config)
    vad_engine = _create_vad_engine(config, model_manager)
    asr_engine = _create_asr_engine(config, model_manager)
    mt_engine = _create_mt_engine(config, model_manager)
    ui_renderer = _create_ui_renderer(config)

    logger.info(
        "engines_created",
        audio=type(audio_source).__name__,
        vad=type(vad_engine).__name__,
        asr=type(asr_engine).__name__,
        mt=type(mt_engine).__name__,
        ui=type(ui_renderer).__name__,
    )

    # --- Assemble Pipeline (Orchestrator receives only interfaces) ---
    pipeline = PipelineOrchestrator(
        audio_source=audio_source,
        vad_engine=vad_engine,
        asr_engine=asr_engine,
        mt_engine=mt_engine,
        ui_renderer=ui_renderer,
        config=config,
    )

    # --- Setup hotkeys ---
    loop = asyncio.get_running_loop()
    from translator.ui.hotkeys import HotkeyManager

    hotkeys = HotkeyManager(config.hotkeys, loop=loop)
    hotkeys.register("toggle_pause", lambda: asyncio.ensure_future(pipeline.toggle_pause()))
    hotkeys.register("quit", lambda: asyncio.ensure_future(pipeline.stop()))
    hotkeys.start()

    # --- Setup system tray ---
    from translator.ui.tray import SystemTray
    import threading

    def _on_quit():
        # Schedule pipeline stop from another thread safely
        asyncio.run_coroutine_threadsafe(pipeline.stop(), loop)

    def _on_settings():
        if type(ui_renderer).__name__ == "TkinterOverlayRenderer":
            asyncio.run_coroutine_threadsafe(ui_renderer.open_settings(), loop)

    def _on_pause():
        asyncio.run_coroutine_threadsafe(pipeline.toggle_pause(), loop)

    tray = SystemTray(on_quit=_on_quit, on_settings=_on_settings, on_pause=_on_pause)
    tray_thread = threading.Thread(target=tray.start, daemon=True, name="system-tray")
    tray_thread.start()

    # --- Setup signal handlers for graceful shutdown ---
    def _signal_handler() -> None:
        logger.info("signal_received", signal="SIGINT/SIGTERM")
        asyncio.ensure_future(pipeline.stop())

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

    # --- Start and run pipeline ---
    try:
        await pipeline.start()
        await pipeline.run()
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt")
    finally:
        await pipeline.stop()
        tray.stop()
        hotkeys.stop()
        logger.info("app_stopped")


def main() -> None:
    """CLI entry point: parse args, load config, run pipeline."""
    args = _parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        print("Create one from config.yaml.example or use --mock", file=sys.stderr)
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

    # Run the async pipeline
    try:
        asyncio.run(_async_main(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

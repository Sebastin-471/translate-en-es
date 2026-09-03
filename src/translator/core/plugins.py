"""Plugin Registry for Engine Implementations.

Allows dynamic registration and discovery of engine implementations
without modifying the composition root (app.py).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog

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


class EngineRegistry:
    """Central registry for engine implementations.

    Engines can be registered at import time (e.g., in their module's
    __init__.py) and retrieved by name from the composition root.
    """

    _audio_sources: dict[str, Callable[[Any], AudioSource]] = {}
    _vad_engines: dict[str, Callable[[Any, ModelManager], VADEngine]] = {}
    _asr_engines: dict[str, Callable[[Any, ModelManager], ASREngine]] = {}
    _mt_engines: dict[str, Callable[[Any, ModelManager], MTEngine]] = {}
    _ui_renderers: dict[str, Callable[[Any], UIRenderer]] = {}

    @classmethod
    def register_audio_source(
        cls,
        name: str,
        factory: Callable[[Any], AudioSource],
        overwrite: bool = False,
    ) -> None:
        """Register an AudioSource implementation.

        Args:
            name: Unique name (e.g., "wasapi", "pipewire", "file").
            factory: Callable that takes AudioConfig and returns AudioSource.
            overwrite: Allow overwriting existing registration.
        """
        if name in cls._audio_sources and not overwrite:
            raise ValueError(f"AudioSource '{name}' already registered")
        cls._audio_sources[name] = factory
        logger.debug("audio_source_registered", name=name)

    @classmethod
    def register_vad_engine(
        cls,
        name: str,
        factory: Callable[[Any, ModelManager], VADEngine],
        overwrite: bool = False,
    ) -> None:
        """Register a VADEngine implementation."""
        if name in cls._vad_engines and not overwrite:
            raise ValueError(f"VADEngine '{name}' already registered")
        cls._vad_engines[name] = factory
        logger.debug("vad_engine_registered", name=name)

    @classmethod
    def register_asr_engine(
        cls,
        name: str,
        factory: Callable[[Any, ModelManager], ASREngine],
        overwrite: bool = False,
    ) -> None:
        """Register an ASREngine implementation."""
        if name in cls._asr_engines and not overwrite:
            raise ValueError(f"ASREngine '{name}' already registered")
        cls._asr_engines[name] = factory
        logger.debug("asr_engine_registered", name=name)

    @classmethod
    def register_mt_engine(
        cls,
        name: str,
        factory: Callable[[Any, ModelManager], MTEngine],
        overwrite: bool = False,
    ) -> None:
        """Register an MTEngine implementation."""
        if name in cls._mt_engines and not overwrite:
            raise ValueError(f"MTEngine '{name}' already registered")
        cls._mt_engines[name] = factory
        logger.debug("mt_engine_registered", name=name)

    @classmethod
    def register_ui_renderer(
        cls,
        name: str,
        factory: Callable[[Any], UIRenderer],
        overwrite: bool = False,
    ) -> None:
        """Register a UIRenderer implementation."""
        if name in cls._ui_renderers and not overwrite:
            raise ValueError(f"UIRenderer '{name}' already registered")
        cls._ui_renderers[name] = factory
        logger.debug("ui_renderer_registered", name=name)

    @classmethod
    def get_audio_source(cls, name: str) -> Callable[[Any], AudioSource]:
        """Get AudioSource factory by name."""
        if name not in cls._audio_sources:
            raise KeyError(
                f"AudioSource '{name}' not registered. "
                f"Available: {list(cls._audio_sources.keys())}"
            )
        return cls._audio_sources[name]

    @classmethod
    def get_vad_engine(cls, name: str) -> Callable[[Any, ModelManager], VADEngine]:
        """Get VADEngine factory by name."""
        if name not in cls._vad_engines:
            raise KeyError(
                f"VADEngine '{name}' not registered. "
                f"Available: {list(cls._vad_engines.keys())}"
            )
        return cls._vad_engines[name]

    @classmethod
    def get_asr_engine(cls, name: str) -> Callable[[Any, ModelManager], ASREngine]:
        """Get ASREngine factory by name."""
        if name not in cls._asr_engines:
            raise KeyError(
                f"ASREngine '{name}' not registered. "
                f"Available: {list(cls._asr_engines.keys())}"
            )
        return cls._asr_engines[name]

    @classmethod
    def get_mt_engine(cls, name: str) -> Callable[[Any, ModelManager], MTEngine]:
        """Get MTEngine factory by name."""
        if name not in cls._mt_engines:
            raise KeyError(
                f"MTEngine '{name}' not registered. "
                f"Available: {list(cls._mt_engines.keys())}"
            )
        return cls._mt_engines[name]

    @classmethod
    def get_ui_renderer(cls, name: str) -> Callable[[Any], UIRenderer]:
        """Get UIRenderer factory by name."""
        if name not in cls._ui_renderers:
            raise KeyError(
                f"UIRenderer '{name}' not registered. "
                f"Available: {list(cls._ui_renderers.keys())}"
            )
        return cls._ui_renderers[name]

    @classmethod
    def list_audio_sources(cls) -> list[str]:
        """List registered audio source names."""
        return list(cls._audio_sources.keys())

    @classmethod
    def list_vad_engines(cls) -> list[str]:
        """List registered VAD engine names."""
        return list(cls._vad_engines.keys())

    @classmethod
    def list_asr_engines(cls) -> list[str]:
        """List registered ASR engine names."""
        return list(cls._asr_engines.keys())

    @classmethod
    def list_mt_engines(cls) -> list[str]:
        """List registered MT engine names."""
        return list(cls._mt_engines.keys())

    @classmethod
    def list_ui_renderers(cls) -> list[str]:
        """List registered UI renderer names."""
        return list(cls._ui_renderers.keys())


def register_builtin_engines() -> None:
    """Register all built-in engine implementations.

    This should be called once at application startup (in app.py).
    """
    # Audio sources
    from translator.infrastructure.audio.wasapi_source import WASAPIAudioSource
    from translator.infrastructure.audio.pipewire_source import PipeWireAudioSource
    from translator.infrastructure.audio.file_source import FileAudioSource
    from translator.infrastructure.mock.mock_audio import MockAudioSource

    EngineRegistry.register_audio_source("wasapi", lambda cfg: WASAPIAudioSource(cfg))
    EngineRegistry.register_audio_source("pipewire", lambda cfg: PipeWireAudioSource(cfg))
    EngineRegistry.register_audio_source("file", lambda cfg: FileAudioSource(cfg))
    EngineRegistry.register_audio_source("mock", lambda cfg: MockAudioSource(cfg, generate_tone=True))

    # VAD engines
    from translator.infrastructure.vad.silero_vad import SileroVADEngine
    from translator.infrastructure.mock.mock_vad import MockVADEngine

    EngineRegistry.register_vad_engine("silero", lambda cfg, mm: SileroVADEngine(cfg, mm))
    EngineRegistry.register_vad_engine("mock", lambda cfg, mm: MockVADEngine(delay_ms=5.0))

    # ASR engines
    from translator.infrastructure.asr.whisper_asr import WhisperASREngine
    from translator.infrastructure.mock.mock_asr import MockASREngine

    EngineRegistry.register_asr_engine("whisper", lambda cfg, mm: WhisperASREngine(cfg, mm))
    EngineRegistry.register_asr_engine("mock", lambda cfg, mm: MockASREngine(
        delay_ms=100.0, language=cfg.language,
    ))

    # MT engines
    from translator.infrastructure.mt.marian_mt import MarianMTEngine
    from translator.infrastructure.mock.mock_mt import MockMTEngine

    EngineRegistry.register_mt_engine("marian", lambda cfg, mm: MarianMTEngine(cfg, mm))
    EngineRegistry.register_mt_engine("mock", lambda cfg, mm: MockMTEngine(
        delay_ms=50.0,
        source_language=cfg.source_language,
        target_language=cfg.target_language,
    ))

    # UI renderers
    from translator.ui.overlay import TkinterOverlayRenderer
    from translator.infrastructure.mock.mock_ui import MockUIRenderer

    EngineRegistry.register_ui_renderer("tkinter", lambda cfg: TkinterOverlayRenderer(cfg))
    EngineRegistry.register_ui_renderer("mock", lambda cfg: MockUIRenderer(show_original=cfg.show_original))

    logger.info("builtin_engines_registered",
                audio_sources=EngineRegistry.list_audio_sources(),
                vad_engines=EngineRegistry.list_vad_engines(),
                asr_engines=EngineRegistry.list_asr_engines(),
                mt_engines=EngineRegistry.list_mt_engines(),
                ui_renderers=EngineRegistry.list_ui_renderers())
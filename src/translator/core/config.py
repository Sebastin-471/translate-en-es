"""Application configuration dataclasses.

All configurable parameters are represented as plain dataclasses with
defaults matching the values in config.yaml. No external dependency
on YAML or any config library — this module defines the *shape* of
configuration only.

The config/ layer is responsible for parsing YAML/TOML/env-vars and
producing an AppConfig instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    """Audio capture configuration."""

    backend: str = "wasapi"
    sample_rate: int = 16_000
    chunk_duration_ms: int = 32
    channels: int = 1
    file_path: str = ""
    device_name: str = ""

    def __post_init__(self) -> None:
        if self.sample_rate not in (8_000, 16_000, 44_100, 48_000):
            raise ValueError(f"Unsupported sample_rate: {self.sample_rate}")
        if self.channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {self.channels}")
        if self.chunk_duration_ms < 10 or self.chunk_duration_ms > 1000:
            raise ValueError(f"chunk_duration_ms must be 10–1000, got {self.chunk_duration_ms}")
        valid_backends = {"wasapi", "pipewire", "file"}
        if self.backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}, got '{self.backend}'")


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""

    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 700
    speech_pad_ms: int = 200
    max_segment_duration_ms: int = 10_000
    partial_interval_ms: int = 500

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}")
        if self.min_speech_duration_ms < 0:
            raise ValueError("min_speech_duration_ms must be non-negative")
        if self.max_segment_duration_ms < 1000:
            raise ValueError("max_segment_duration_ms must be ≥ 1000")


@dataclass
class ASRConfig:
    """Automatic Speech Recognition configuration."""

    model_size: str = "large-v3-turbo"
    device: str = "auto"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 5
    model_path: str = ""
    cpu_threads: int = 4

    def __post_init__(self) -> None:
        valid_sizes = {"tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"}
        if self.model_size not in valid_sizes:
            raise ValueError(f"model_size must be one of {valid_sizes}, got '{self.model_size}'")
        valid_devices = {"auto", "cuda", "cpu"}
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got '{self.device}'")
        valid_compute = {"int8", "float16", "float32"}
        if self.compute_type not in valid_compute:
            raise ValueError(
                f"compute_type must be one of {valid_compute}, got '{self.compute_type}'"
            )


@dataclass
class MTConfig:
    """Machine Translation configuration."""

    model_name: str = "Helsinki-NLP/opus-mt-en-es"
    model_path: str = ""
    source_language: str = "en"
    target_language: str = "es"
    device: str = "auto"
    compute_type: str = "int8"
    beam_size: int = 4

    def __post_init__(self) -> None:
        valid_devices = {"auto", "cuda", "cpu"}
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got '{self.device}'")
        if not self.source_language:
            raise ValueError("source_language must not be empty")
        if not self.target_language:
            raise ValueError("target_language must not be empty")


@dataclass
class GPUConfig:
    """GPU / Model Manager configuration."""

    max_vram_mb: int = 0
    preload_models: bool = True
    device_index: int = 0


@dataclass
class UIConfig:
    """UI Overlay configuration."""

    enabled: bool = True
    position: str = "bottom"
    margin_x: int = 50
    margin_y: int = 80
    font_family: str = "Segoe UI"
    font_size: int = 22
    text_color: str = "#FFFFFF"
    background_color: str = "#000000"
    background_opacity: float = 0.75
    max_lines: int = 3
    fade_after_seconds: float = 8.0
    show_original: bool = True

    def __post_init__(self) -> None:
        valid_positions = {"bottom", "top", "center"}
        if self.position not in valid_positions:
            raise ValueError(
                f"position must be one of {valid_positions}, got '{self.position}'"
            )
        if not 0.0 <= self.background_opacity <= 1.0:
            raise ValueError(
                f"background_opacity must be 0.0–1.0, got {self.background_opacity}"
            )


@dataclass
class HotkeyConfig:
    """Global hotkey configuration."""

    toggle_overlay: str = "ctrl+shift+t"
    toggle_pause: str = "ctrl+shift+p"
    quit: str = "ctrl+shift+q"


@dataclass
class LoggingConfig:
    """Logging and observability configuration."""

    level: str = "INFO"
    format: str = "console"
    file_path: str = ""
    log_latencies: bool = True

    def __post_init__(self) -> None:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}, got '{self.level}'")
        valid_formats = {"json", "console"}
        if self.format not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}, got '{self.format}'")


@dataclass
class PipelineConfig:
    """Pipeline orchestration configuration."""

    queue_size: int = 50
    mock_mode: bool = False
    mock_delay_ms: int = 100

    def __post_init__(self) -> None:
        if self.queue_size < 0:
            raise ValueError(f"queue_size must be non-negative, got {self.queue_size}")


@dataclass
class ConfigConfig:
    """Configuration system settings."""

    watch_for_changes: bool = True
    watch_interval_ms: int = 500


@dataclass
class AppConfig:
    """Top-level application configuration.

    Aggregates all sub-configurations into a single root object.
    The composition root reads this to decide which concrete
    implementations to instantiate.
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    mt: MTConfig = field(default_factory=MTConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    config: ConfigConfig = field(default_factory=ConfigConfig)

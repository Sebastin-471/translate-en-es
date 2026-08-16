"""Configuration loader: YAML file → validated AppConfig.

Reads a YAML configuration file and maps it to the AppConfig dataclass
hierarchy with full validation. Supports environment variable overrides
using the TRANSLATOR_ prefix with double-underscore nesting.

Examples of env var overrides:
  TRANSLATOR_ASR__MODEL_SIZE=medium
  TRANSLATOR_ASR__DEVICE=cpu
  TRANSLATOR_PIPELINE__MOCK_MODE=true
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog
import yaml  # type: ignore[import-untyped]

from translator.core.config import (
    AppConfig,
    ASRConfig,
    AudioConfig,
    GPUConfig,
    HotkeyConfig,
    LoggingConfig,
    MTConfig,
    PipelineConfig,
    UIConfig,
    VADConfig,
)

logger = structlog.get_logger(__name__)

_ENV_PREFIX = "TRANSLATOR_"

# Mapping from top-level YAML keys to their config dataclass
_SECTION_MAP: dict[str, type[Any]] = {
    "audio": AudioConfig,
    "vad": VADConfig,
    "asr": ASRConfig,
    "mt": MTConfig,
    "gpu": GPUConfig,
    "ui": UIConfig,
    "hotkeys": HotkeyConfig,
    "logging": LoggingConfig,
    "pipeline": PipelineConfig,
}


def load_config(path: str | Path) -> AppConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A fully validated AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file contains invalid values.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Apply environment variable overrides
    raw = _apply_env_overrides(raw)

    # Build section configs
    sections: dict[str, Any] = {}
    for section_name, config_cls in _SECTION_MAP.items():
        section_data = raw.get(section_name, {})
        if not isinstance(section_data, dict):
            section_data = {}

        # Filter to only fields that exist in the dataclass
        valid_fields = {f.name for f in config_cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in section_data.items() if k in valid_fields}

        try:
            sections[section_name] = config_cls(**filtered)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid configuration in section '{section_name}': {e}"
            ) from e

    config = AppConfig(**sections)

    logger.info(
        "config_loaded",
        path=str(config_path),
        audio_backend=config.audio.backend,
        asr_model=config.asr.model_size,
        asr_device=config.asr.device,
        mt_model=config.mt.model_name,
        mock_mode=config.pipeline.mock_mode,
    )

    return config


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply TRANSLATOR_ prefixed environment variables as overrides.

    Environment variables use double underscores for nesting:
      TRANSLATOR_ASR__MODEL_SIZE=medium → raw["asr"]["model_size"] = "medium"
      TRANSLATOR_PIPELINE__MOCK_MODE=true → raw["pipeline"]["mock_mode"] = True
    """
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue

        # Strip prefix and split on double underscores
        rest = key[len(_ENV_PREFIX):].lower()
        parts = rest.split("__")

        if len(parts) == 2:
            section, field = parts
            if section not in raw:
                raw[section] = {}
            raw[section][field] = _coerce_value(value)
            logger.debug(
                "config_env_override",
                env_var=key,
                section=section,
                field=field,
                value=value,
            )

    return raw


def _coerce_value(value: str) -> Any:
    """Coerce a string env var value to the most appropriate Python type."""
    # Booleans
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # Integers
    try:
        return int(value)
    except ValueError:
        pass

    # Floats
    try:
        return float(value)
    except ValueError:
        pass

    # String
    return value

"""Configuration loader: YAML file → validated AppConfig.

Reads a YAML configuration file and maps it to the AppConfig dataclass
hierarchy with full validation. Supports environment variable overrides
using the TRANSLATOR_ prefix with double-underscore nesting.

Examples of env var overrides:
  TRANSLATOR_ASR__MODEL_SIZE=medium
  TRANSLATOR_ASR__DEVICE=cpu
  TRANSLATOR_PIPELINE__MOCK_MODE=true

Supports environment-specific configs:
  - config/base.yaml (always loaded)
  - config/development.yaml (loaded when TRANSLATOR_ENV=development)
  - config/production.yaml (loaded when TRANSLATOR_ENV=production)

Hot-reload support via ConfigWatcher (uses watchfiles).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import yaml  # type: ignore[import-untyped]

from translator.core.config import (
    AppConfig,
    ASRConfig,
    AudioConfig,
    ConfigConfig,
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
_DEFAULT_ENV = "development"

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
    "config": ConfigConfig,
}


@dataclass(frozen=True)
class ConfigChangeEvent:
    """Event emitted when configuration changes (hot-reload)."""
    old_config: AppConfig
    new_config: AppConfig
    changed_sections: set[str]


class ConfigWatcher:
    """Watches configuration files for changes and emits ConfigChangeEvent.

    Uses watchfiles for cross-platform file watching.
    """

    def __init__(
        self,
        config_paths: list[Path],
        callback: Callable[[ConfigChangeEvent], None],
        interval_ms: int = 500,
    ) -> None:
        self._config_paths = config_paths
        self._callback = callback
        self._interval_ms = interval_ms
        self._task: Any = None
        self._running = False

    async def start(self, current_config: AppConfig) -> None:
        """Start watching for config changes."""
        self._running = True
        self._current_config = current_config
        try:
            import watchfiles
            self._watcher = watchfiles.awatch(
                *self._config_paths,
                debounce=self._interval_ms / 1000,
            )
            self._task = __import__("asyncio").create_task(self._watch_loop())
            logger.info("config_watcher_started", paths=[str(p) for p in self._config_paths])
        except ImportError:
            logger.warning("watchfiles_not_installed_config_watcher_disabled")

    async def stop(self) -> None:
        """Stop watching for config changes."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except __import__("asyncio").CancelledError:
                pass
        logger.info("config_watcher_stopped")

    async def _watch_loop(self) -> None:
        """Watch loop that reloads config on changes."""
        async for changes in self._watcher:
            if not self._running:
                break
            logger.info("config_file_changed", changes=changes)
            try:
                new_config = load_config(self._config_paths[0])  # Main config path
                changed = self._detect_changes(self._current_config, new_config)
                if changed:
                    event = ConfigChangeEvent(
                        old_config=self._current_config,
                        new_config=new_config,
                        changed_sections=changed,
                    )
                    self._current_config = new_config
                    self._callback(event)
                    logger.info("config_reloaded", changed_sections=list(changed))
            except Exception as e:
                logger.exception("config_reload_failed", error=str(e))

    def _detect_changes(self, old: AppConfig, new: AppConfig) -> set[str]:
        """Detect which config sections changed."""
        changed = set()
        for section_name in _SECTION_MAP:
            old_section = getattr(old, section_name)
            new_section = getattr(new, section_name)
            if old_section != new_section:
                changed.add(section_name)
        return changed


def load_config(path: str | Path, env: str | None = None) -> AppConfig:
    """Load and validate configuration from YAML files.

    Loads base.yaml first, then overlays environment-specific config
    (development.yaml, production.yaml, etc.) based on TRANSLATOR_ENV
    or the `env` parameter.

    Args:
        path: Path to the main configuration file (typically config.yaml or base.yaml).
        env: Optional environment name. If not provided, reads TRANSLATOR_ENV
             environment variable (defaults to "development").

    Returns:
        A fully validated AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file contains invalid values.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Determine environment
    if env is None:
        env = os.environ.get("TRANSLATOR_ENV", _DEFAULT_ENV)

    # Load base config
    base_path = config_path.parent / "base.yaml"
    if not base_path.exists():
        # Fallback: treat given path as base if no base.yaml exists
        base_path = config_path

    with open(base_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Load environment-specific overlay
    env_path = config_path.parent / f"{env}.yaml"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            env_raw: dict[str, Any] = yaml.safe_load(f) or {}
        raw = _deep_merge(raw, env_raw)
        logger.info("config_env_loaded", env=env, path=str(env_path))
    else:
        logger.warning("config_env_not_found", env=env, expected_path=str(env_path))

    # Apply environment variable overrides (highest priority)
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
        env=env,
        audio_backend=config.audio.backend,
        asr_model=config.asr.model_size,
        asr_device=config.asr.device,
        mt_model=config.mt.model_name,
        mock_mode=config.pipeline.mock_mode,
    )

    return config


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries (overlay takes precedence)."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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

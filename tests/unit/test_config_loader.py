"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from translator.config.loader import load_config
from translator.core.config import (
    AppConfig,
    ASRConfig,
    AudioConfig,
    MTConfig,
    PipelineConfig,
    VADConfig,
)


class TestAppConfig:
    """Tests for AppConfig dataclass validation."""

    def test_default_config_is_valid(self) -> None:
        config = AppConfig()
        assert config.audio.sample_rate == 16000
        assert config.asr.model_size == "large-v3-turbo"
        assert config.mt.source_language == "en"
        assert config.mt.target_language == "es"

    def test_invalid_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="Unsupported sample_rate"):
            AudioConfig(sample_rate=22050)

    def test_invalid_backend(self) -> None:
        with pytest.raises(ValueError, match="backend must be one of"):
            AudioConfig(backend="alsa")

    def test_invalid_vad_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            VADConfig(threshold=1.5)

    def test_invalid_asr_model(self) -> None:
        with pytest.raises(ValueError, match="model_size must be one of"):
            ASRConfig(model_size="huge")

    def test_invalid_mt_device(self) -> None:
        with pytest.raises(ValueError, match="device must be one of"):
            MTConfig(device="tpu")

    def test_empty_source_language(self) -> None:
        with pytest.raises(ValueError, match="source_language must not be empty"):
            MTConfig(source_language="")

    def test_negative_queue_size(self) -> None:
        with pytest.raises(ValueError, match="queue_size must be non-negative"):
            PipelineConfig(queue_size=-1)


class TestConfigLoader:
    """Tests for YAML config loading."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        config_data = {
            "audio": {"backend": "file", "sample_rate": 16000},
            "asr": {"model_size": "small", "device": "cpu"},
            "pipeline": {"mock_mode": True},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        config = load_config(config_file)
        assert config.audio.backend == "file"
        assert config.asr.model_size == "small"
        assert config.pipeline.mock_mode is True

    def test_load_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_empty_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("", encoding="utf-8")

        config = load_config(config_file)
        # Should use all defaults
        assert config.audio.sample_rate == 16000
        assert config.pipeline.mock_mode is False

    def test_load_partial_config(self, tmp_path: Path) -> None:
        config_data = {"asr": {"model_size": "medium"}}
        config_file = tmp_path / "partial.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        config = load_config(config_file)
        assert config.asr.model_size == "medium"
        # Other sections should have defaults
        assert config.audio.backend == "wasapi"
        assert config.vad.threshold == 0.5

    def test_env_var_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"asr": {"model_size": "large-v3"}}), encoding="utf-8")

        monkeypatch.setenv("TRANSLATOR_ASR__MODEL_SIZE", "small")
        monkeypatch.setenv("TRANSLATOR_PIPELINE__MOCK_MODE", "true")

        config = load_config(config_file)
        assert config.asr.model_size == "small"
        assert config.pipeline.mock_mode is True

    def test_invalid_section_value(self, tmp_path: Path) -> None:
        config_data = {"audio": {"sample_rate": 22050}}
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid configuration"):
            load_config(config_file)

    def test_unknown_keys_ignored(self, tmp_path: Path) -> None:
        config_data = {"audio": {"backend": "file", "unknown_key": "ignored"}}
        config_file = tmp_path / "extra.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        config = load_config(config_file)
        assert config.audio.backend == "file"

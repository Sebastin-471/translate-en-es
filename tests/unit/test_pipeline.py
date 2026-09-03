"""Tests for pipeline orchestration with mock engines."""

from __future__ import annotations

import asyncio

import pytest

from translator.core.config import AppConfig, AudioConfig, LoggingConfig, PipelineConfig
from translator.core.events import TranslationResult
from translator.infrastructure.mock.mock_asr import MockASREngine
from translator.infrastructure.mock.mock_audio import MockAudioSource
from translator.infrastructure.mock.mock_mt import MockMTEngine
from translator.infrastructure.mock.mock_ui import MockUIRenderer
from translator.infrastructure.mock.mock_vad import MockVADEngine
from translator.pipeline.orchestrator import PipelineOrchestrator


class TestPipelineOrchestrator:
    """Tests for the PipelineOrchestrator with mock engines."""

    def _make_config(self) -> AppConfig:
        """Create a minimal test config."""
        config = AppConfig()
        config.audio = AudioConfig(backend="file", sample_rate=16000, chunk_duration_ms=30)
        config.pipeline = PipelineConfig(queue_size=10, mock_mode=True, mock_delay_ms=5)
        config.logging = LoggingConfig(level="DEBUG", format="console", log_latencies=True)
        return config

    @pytest.mark.asyncio
    async def test_pipeline_starts_and_stops(self) -> None:
        """Pipeline should start and stop cleanly with mock engines."""
        config = self._make_config()

        audio = MockAudioSource(config.audio)
        vad = MockVADEngine(chunks_per_segment=3, delay_ms=0)
        asr = MockASREngine(delay_ms=1)
        mt = MockMTEngine(delay_ms=1)
        ui = MockUIRenderer()

        pipeline = PipelineOrchestrator(
            audio_source=audio,
            vad_engine=vad,
            asr_engine=asr,
            mt_engine=mt,
            ui_renderer=ui,
            config=config,
        )

        await pipeline.start()

        # Let it run for a short time to process a few chunks
        await asyncio.sleep(0.5)

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_produces_translations(self) -> None:
        """Mock pipeline should produce TranslationResults in the UI."""
        config = self._make_config()

        audio = MockAudioSource(config.audio)
        vad = MockVADEngine(chunks_per_segment=3, delay_ms=0)
        asr = MockASREngine(delay_ms=1)
        mt = MockMTEngine(delay_ms=1)
        ui = MockUIRenderer()

        pipeline = PipelineOrchestrator(
            audio_source=audio,
            vad_engine=vad,
            asr_engine=asr,
            mt_engine=mt,
            ui_renderer=ui,
            config=config,
        )

        await pipeline.start()

        # Wait enough time for at least one full cycle
        # 3 chunks × 30ms = 90ms for VAD, plus processing
        await asyncio.sleep(1.0)

        await pipeline.stop()

        # The UI should have received at least one translation
        assert len(ui.translation_history) > 0

        first = ui.translation_history[0]
        assert isinstance(first, TranslationResult)
        assert first.original_text  # Non-empty
        assert first.translated_text  # Non-empty

    @pytest.mark.asyncio
    async def test_pipeline_metrics_recorded(self) -> None:
        """Pipeline metrics should have data after processing."""
        config = self._make_config()

        audio = MockAudioSource(config.audio)
        vad = MockVADEngine(chunks_per_segment=3, delay_ms=0)
        asr = MockASREngine(delay_ms=1)
        mt = MockMTEngine(delay_ms=1)
        ui = MockUIRenderer()

        pipeline = PipelineOrchestrator(
            audio_source=audio,
            vad_engine=vad,
            asr_engine=asr,
            mt_engine=mt,
            ui_renderer=ui,
            config=config,
        )

        await pipeline.start()
        await asyncio.sleep(1.0)
        await pipeline.stop()

        summaries = pipeline.metrics.get_all_summaries()
        # Should have metrics for at least VAD stage
        assert "vad" in summaries
        assert summaries["vad"]["count"] > 0

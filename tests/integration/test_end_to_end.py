"""End-to-end integration test for the full translation pipeline.

Runs the complete pipeline (AudioSource → VAD → ASR → MT → UI) with
mock engines to verify that data flows correctly through all stages
and that the orchestrator manages the lifecycle properly.

This test does NOT require a GPU or any ML models — it uses the mock
implementations throughout. For tests with real models, see the
`@pytest.mark.gpu` marked tests.
"""

from __future__ import annotations

import asyncio

import pytest

from translator.core.config import (
    AppConfig,
    AudioConfig,
    LoggingConfig,
    PipelineConfig,
)
from translator.core.events import TranslationResult
from translator.infrastructure.mock.mock_asr import MockASREngine
from translator.infrastructure.mock.mock_audio import MockAudioSource
from translator.infrastructure.mock.mock_mt import MockMTEngine
from translator.infrastructure.mock.mock_ui import MockUIRenderer
from translator.infrastructure.mock.mock_vad import MockVADEngine
from translator.pipeline.orchestrator import PipelineOrchestrator


class TestEndToEnd:
    """End-to-end pipeline integration tests."""

    def _make_config(self) -> AppConfig:
        """Create test config with fast mock settings."""
        config = AppConfig()
        config.audio = AudioConfig(
            backend="file",
            sample_rate=16000,
            chunk_duration_ms=30,
        )
        config.pipeline = PipelineConfig(
            queue_size=20,
            mock_mode=True,
            mock_delay_ms=5,
        )
        config.logging = LoggingConfig(
            level="DEBUG",
            format="console",
            log_latencies=True,
        )
        return config

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_pipeline_mock_mode(self) -> None:
        """Full pipeline should produce translations from mock audio.

        This verifies:
          1. AudioSource emits chunks to the audio queue.
          2. VAD accumulates chunks and emits segments.
          3. ASR produces transcripts from segments.
          4. MT translates transcripts.
          5. UI receives translation results.
          6. Pipeline shuts down cleanly.
        """
        config = self._make_config()

        # Create all mock engines
        audio = MockAudioSource(config.audio, generate_tone=True)
        vad = MockVADEngine(chunks_per_segment=5, delay_ms=0)
        asr = MockASREngine(delay_ms=5, language="en")
        mt = MockMTEngine(delay_ms=3, source_language="en", target_language="es")
        ui = MockUIRenderer(show_original=True)

        # Assemble pipeline (the orchestrator only sees interfaces)
        pipeline = PipelineOrchestrator(
            audio_source=audio,
            vad_engine=vad,
            asr_engine=asr,
            mt_engine=mt,
            ui_renderer=ui,
            config=config,
        )

        # Start and run
        await pipeline.start()
        await asyncio.sleep(2.0)  # Let it process for 2 seconds
        await pipeline.stop()

        # Verify translations were produced
        history = ui.translation_history
        assert len(history) > 0, "Pipeline produced no translations"

        # Verify each translation has the correct structure
        for translation in history:
            assert isinstance(translation, TranslationResult)
            assert translation.original_text, "original_text is empty"
            assert translation.translated_text, "translated_text is empty"
            assert translation.source_language == "en"
            assert translation.target_language == "es"
            assert translation.sequence_id, "sequence_id is empty"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_metrics_end_to_end(self) -> None:
        """Pipeline should collect meaningful metrics across all stages."""
        config = self._make_config()

        audio = MockAudioSource(config.audio)
        vad = MockVADEngine(chunks_per_segment=3, delay_ms=0)
        asr = MockASREngine(delay_ms=2)
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
        await asyncio.sleep(1.5)
        await pipeline.stop()

        summaries = pipeline.metrics.get_all_summaries()

        # Should have metrics for at least the VAD stage
        assert "vad" in summaries
        assert summaries["vad"]["count"] > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_graceful_shutdown(self) -> None:
        """Pipeline should shut down cleanly without hanging."""
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
        await asyncio.sleep(0.3)

        # Stop should complete within the timeout
        try:
            await asyncio.wait_for(pipeline.stop(), timeout=15.0)
        except TimeoutError:
            pytest.fail("Pipeline shutdown timed out after 15 seconds")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sequence_id_preserved_end_to_end(self) -> None:
        """Sequence IDs should be traceable from VAD through to UI."""
        config = self._make_config()

        audio = MockAudioSource(config.audio, generate_tone=True)
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
        await asyncio.sleep(1.5)
        await pipeline.stop()

        # Every translation should have a non-empty sequence_id
        for translation in ui.translation_history:
            assert translation.sequence_id, "Missing sequence_id in translation"
            assert len(translation.sequence_id) > 0

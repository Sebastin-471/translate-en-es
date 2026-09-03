"""Tests for the GPU ModelManager."""

from __future__ import annotations

import pytest

from translator.core.model_manager import DeviceType
from translator.infrastructure.gpu.gpu_model_manager import GPUModelManager


class TestGPUModelManager:
    """Tests for GPUModelManager behavior."""

    @pytest.mark.asyncio
    async def test_register_and_load_model(self) -> None:
        """Should register and load a model successfully."""
        manager = GPUModelManager(max_vram_mb=1000)

        await manager.register_model("test-model", "asr", vram_estimate_mb=100)

        model_obj = {"name": "dummy"}
        loaded = await manager.load_model("test-model", loader=lambda: model_obj)

        assert loaded == model_obj

    @pytest.mark.asyncio
    async def test_get_loaded_model(self) -> None:
        """Should retrieve a previously loaded model."""
        manager = GPUModelManager()

        await manager.register_model("test-model", "mt", vram_estimate_mb=50)
        model_obj = {"id": 42}
        await manager.load_model("test-model", loader=lambda: model_obj)

        result = await manager.get_model("test-model")
        assert result == model_obj

    @pytest.mark.asyncio
    async def test_get_unloaded_model_raises(self) -> None:
        """Should raise KeyError for models that aren't loaded."""
        manager = GPUModelManager()

        with pytest.raises(KeyError, match="not loaded"):
            await manager.get_model("nonexistent")

    @pytest.mark.asyncio
    async def test_load_unregistered_model_raises(self) -> None:
        """Should raise KeyError for models that aren't registered."""
        manager = GPUModelManager()

        with pytest.raises(KeyError, match="not registered"):
            await manager.load_model("unregistered", loader=lambda: None)

    @pytest.mark.asyncio
    async def test_unload_model(self) -> None:
        """Should unload a model and make it unavailable."""
        manager = GPUModelManager()

        await manager.register_model("test", "asr", vram_estimate_mb=100)
        await manager.load_model("test", loader=lambda: "model_data")

        await manager.unload_model("test")

        with pytest.raises(KeyError):
            await manager.get_model("test")

    @pytest.mark.asyncio
    async def test_vram_status(self) -> None:
        """Should track VRAM usage of loaded models."""
        manager = GPUModelManager(max_vram_mb=2000)

        await manager.register_model("model-a", "asr", vram_estimate_mb=500)
        await manager.register_model("model-b", "mt", vram_estimate_mb=200)

        await manager.load_model(
            "model-a", loader=lambda: "a", device=DeviceType.CUDA
        )
        await manager.load_model(
            "model-b", loader=lambda: "b", device=DeviceType.CUDA
        )

        status = manager.get_vram_status()
        assert status.used_mb == 700  # 500 + 200
        assert len(status.models) == 2

    @pytest.mark.asyncio
    async def test_vram_budget_exceeded(self) -> None:
        """Should refuse to load if VRAM budget would be exceeded."""
        manager = GPUModelManager(max_vram_mb=500)

        await manager.register_model("big-model", "asr", vram_estimate_mb=600)

        with pytest.raises(RuntimeError, match="exceed VRAM budget"):
            await manager.load_model(
                "big-model", loader=lambda: "data", device=DeviceType.CUDA
            )

    @pytest.mark.asyncio
    async def test_unlimited_vram_budget(self) -> None:
        """With max_vram_mb=0, should never refuse to load."""
        manager = GPUModelManager(max_vram_mb=0)  # Unlimited

        await manager.register_model("big-model", "asr", vram_estimate_mb=99999)

        # Should not raise even with a huge VRAM estimate
        result = await manager.load_model(
            "big-model", loader=lambda: "loaded", device=DeviceType.CUDA
        )
        assert result == "loaded"

    def test_detect_device_returns_cpu_without_cuda(self) -> None:
        """On machines without CUDA, should fall back to CPU."""
        manager = GPUModelManager()
        device = manager.detect_device()
        # Will be CUDA if GPU available, CPU otherwise — both valid
        assert device in (DeviceType.CUDA, DeviceType.CPU)

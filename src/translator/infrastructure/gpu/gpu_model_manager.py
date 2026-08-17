"""GPUModelManager: Centralized GPU resource and model lifecycle management.

Tracks loaded models, monitors VRAM usage, enforces budget limits,
and supports hot-reload of models. Uses torch.cuda for VRAM tracking
when available, with graceful fallback for CPU-only environments.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import structlog

from translator.core.model_manager import DeviceType, ModelInfo, VRAMStatus

logger = structlog.get_logger(__name__)


class GPUModelManager:
    """Concrete ModelManager implementation with VRAM tracking.

    This class satisfies the ModelManager Protocol (structural subtyping).

    Manages the lifecycle of all ML models in the pipeline:
      - Registers models with estimated VRAM usage.
      - Enforces a configurable VRAM budget.
      - Provides load/unload operations for hot-swapping.
      - Auto-detects CUDA availability.
    """

    def __init__(self, max_vram_mb: float = 0, device_index: int = 0) -> None:
        """Initialize the model manager.

        Args:
            max_vram_mb: Maximum VRAM budget in MB (0 = unlimited).
            device_index: CUDA device index for multi-GPU systems.
        """
        self._max_vram_mb = max_vram_mb
        self._device_index = device_index
        self._registry: dict[str, ModelInfo] = {}
        self._loaded_models: dict[str, Any] = {}
        self._detected_device: DeviceType | None = None

    async def register_model(
        self, name: str, model_type: str, vram_estimate_mb: float
    ) -> None:
        """Register a model that will be managed."""
        info = ModelInfo(
            name=name,
            model_type=model_type,
            device=self.detect_device(),
            vram_usage_mb=vram_estimate_mb,
            is_loaded=False,
        )
        self._registry[name] = info
        logger.info(
            "model_registered",
            name=name,
            model_type=model_type,
            vram_estimate_mb=vram_estimate_mb,
        )

    async def load_model(
        self, name: str, loader: Any, device: DeviceType | None = None
    ) -> Any:
        """Load a model into memory, checking VRAM budget."""
        if name not in self._registry:
            raise KeyError(f"Model '{name}' not registered. Call register_model() first.")

        if name in self._loaded_models:
            logger.warning("model_already_loaded", name=name)
            return self._loaded_models[name]

        info = self._registry[name]
        target_device = device or info.device

        # Check VRAM budget
        if target_device == DeviceType.CUDA and self._max_vram_mb > 0:
            current_usage = sum(
                m.vram_usage_mb
                for m in self._registry.values()
                if m.is_loaded and m.device == DeviceType.CUDA
            )
            if current_usage + info.vram_usage_mb > self._max_vram_mb:
                raise RuntimeError(
                    f"Loading '{name}' ({info.vram_usage_mb}MB) would exceed VRAM budget: "
                    f"{current_usage}MB used / {self._max_vram_mb}MB max. "
                    f"Unload another model first or increase budget."
                )

        logger.info("model_loading", name=name, device=target_device.value)
        start = time.monotonic()

        # Call the loader callable
        if callable(loader):
            model = loader()
        else:
            model = loader

        elapsed = time.monotonic() - start

        self._loaded_models[name] = model
        self._registry[name] = ModelInfo(
            name=info.name,
            model_type=info.model_type,
            device=target_device,
            vram_usage_mb=info.vram_usage_mb,
            is_loaded=True,
            metadata=info.metadata,
        )

        logger.info(
            "model_loaded",
            name=name,
            device=target_device.value,
            load_time_s=round(elapsed, 2),
            vram_mb=info.vram_usage_mb,
        )

        return model

    async def unload_model(self, name: str) -> None:
        """Unload a model and free its resources."""
        if name not in self._loaded_models:
            logger.warning("model_not_loaded", name=name)
            return

        del self._loaded_models[name]

        info = self._registry[name]
        self._registry[name] = ModelInfo(
            name=info.name,
            model_type=info.model_type,
            device=info.device,
            vram_usage_mb=info.vram_usage_mb,
            is_loaded=False,
            metadata=info.metadata,
        )

        # Try to free GPU memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("model_unloaded", name=name)

    async def get_model(self, name: str) -> Any:
        """Retrieve a loaded model by name."""
        if name not in self._loaded_models:
            raise KeyError(
                f"Model '{name}' is not loaded. "
                f"Available loaded models: {list(self._loaded_models.keys())}"
            )
        return self._loaded_models[name]

    def get_vram_status(self) -> VRAMStatus:
        """Return current VRAM usage summary."""
        total_mb = 0.0
        free_mb = 0.0

        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(self._device_index)
                total_mb = props.total_mem / (1024 * 1024)
                free_mb = (
                    torch.cuda.get_device_properties(self._device_index).total_mem
                    - torch.cuda.memory_allocated(self._device_index)
                ) / (1024 * 1024)
        except (ImportError, RuntimeError):
            pass

        used_mb = sum(
            m.vram_usage_mb
            for m in self._registry.values()
            if m.is_loaded and m.device == DeviceType.CUDA
        )

        loaded_models = [m for m in self._registry.values() if m.is_loaded]

        return VRAMStatus(
            total_mb=total_mb,
            used_mb=used_mb,
            free_mb=max(0.0, free_mb),
            models=loaded_models,
        )

    def detect_device(self) -> DeviceType:
        """Detect and return the best available compute device."""
        if self._detected_device is not None:
            return self._detected_device

        try:
            import torch

            if torch.cuda.is_available():
                self._detected_device = DeviceType.CUDA
                gpu_name = torch.cuda.get_device_name(self._device_index)
                vram_mb = torch.cuda.get_device_properties(self._device_index).total_memory / (
                    1024 * 1024
                )
                logger.info(
                    "gpu_detected",
                    device=gpu_name,
                    vram_mb=round(vram_mb),
                    device_index=self._device_index,
                )
                return DeviceType.CUDA
        except (ImportError, RuntimeError):
            pass

        self._detected_device = DeviceType.CPU
        logger.info("gpu_not_available", fallback="cpu")
        return DeviceType.CPU

"""ModelManager protocol and GPU resource types.

Centralizes loading, unloading, and VRAM tracking for all ML models
used in the pipeline. The composition root creates a single ModelManager
instance and passes it to engines that need GPU resources.

Design rationale:
  - Single point of control for VRAM budget (critical for RTX 3060 12GB).
  - Enables hot-reload of models (e.g., switch ASR model size) without
    restarting the entire application.
  - Engines register themselves with the ModelManager at startup; they
    don't load models independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class DeviceType(Enum):
    """Compute device for model inference."""

    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Metadata about a loaded model.

    Attributes:
        name: Human-readable model name (e.g., "whisper-large-v3-turbo").
        model_type: Category of model ("asr", "mt", "vad").
        device: Device where the model is loaded.
        vram_usage_mb: Estimated VRAM usage in MB (0 for CPU models).
        is_loaded: Whether the model is currently in memory.
        metadata: Additional model-specific metadata.
    """

    name: str
    model_type: str
    device: DeviceType
    vram_usage_mb: float
    is_loaded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VRAMStatus:
    """Current VRAM usage summary.

    Attributes:
        total_mb: Total VRAM available on the device.
        used_mb: VRAM used by models managed by this ModelManager.
        free_mb: VRAM available for additional models.
        models: List of loaded model info.
    """

    total_mb: float
    used_mb: float
    free_mb: float
    models: list[ModelInfo] = field(default_factory=list)


@runtime_checkable
class ModelManager(Protocol):
    """Centralized manager for ML model lifecycle and GPU resources.

    Responsibilities:
      1. Track which models are loaded and their VRAM usage.
      2. Enforce VRAM budget (refuse to load if budget would be exceeded).
      3. Provide load/unload operations for hot-swapping models.
      4. Detect available devices (CUDA vs CPU) and select the best one.

    The composition root creates one ModelManager and injects it into
    engines that need GPU-heavy models.
    """

    async def register_model(self, name: str, model_type: str, vram_estimate_mb: float) -> None:
        """Register a model that will be managed.

        Args:
            name: Unique model identifier.
            model_type: Category ("asr", "mt", "vad").
            vram_estimate_mb: Estimated VRAM usage when loaded.
        """
        ...

    async def load_model(
        self, name: str, loader: Any, device: DeviceType | None = None
    ) -> Any:
        """Load a model into memory.

        Args:
            name: Previously registered model name.
            loader: A callable that loads the model (engine-specific).
            device: Target device (None = auto-select).

        Returns:
            The loaded model object.

        Raises:
            RuntimeError: If VRAM budget would be exceeded.
        """
        ...

    async def unload_model(self, name: str) -> None:
        """Unload a model and free its resources.

        Args:
            name: The model to unload.
        """
        ...

    async def get_model(self, name: str) -> Any:
        """Retrieve a loaded model by name.

        Args:
            name: The model to retrieve.

        Returns:
            The loaded model object.

        Raises:
            KeyError: If the model is not loaded.
        """
        ...

    def get_vram_status(self) -> VRAMStatus:
        """Return current VRAM usage summary."""
        ...

    def detect_device(self) -> DeviceType:
        """Detect and return the best available compute device."""
        ...

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mowerseg.core.backend import InferenceBackend
from mowerseg.core.frame import Frame
from mowerseg.core.result import PerceptionResult
from mowerseg.models.registry import ModelConfig


class PerceptionTask(ABC):
    """Solves one perception problem. Does not know NPU SDK details."""

    name: str = "base"

    def __init__(
        self,
        model_config: ModelConfig,
        backend: InferenceBackend,
        *,
        product_config: dict[str, Any] | None = None,
        frequency_hz: float | None = None,
        enabled: bool = True,
        priority: int = 0,
    ) -> None:
        self.model_config = model_config
        self.backend = backend
        self.product_config = product_config or {}
        self.frequency_hz = frequency_hz
        self.enabled = enabled
        self.priority = priority

    @abstractmethod
    def setup(self) -> None:
        """Load backend artifact and task-specific helpers."""

    @abstractmethod
    def predict(self, frame: Frame) -> PerceptionResult:
        """Preprocess → backend.infer → postprocess → canonical result."""

    def close(self) -> None:
        self.backend.close()

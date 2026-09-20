from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mowerseg.models.registry import ModelConfig


class InferenceBackend(ABC):
    """Hardware / runtime executor. Handles tensors and devices only."""

    name: str = "base"

    @abstractmethod
    def load(self, model_config: ModelConfig) -> None:
        """Load the compiled / ready artifact for this backend."""

    @abstractmethod
    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run forward pass. Inputs/outputs are tensor-like payloads."""

    @abstractmethod
    def close(self) -> None:
        """Release runtime resources."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Human-readable device string, e.g. cpu / cuda:0 / bpu:0."""

    @property
    def is_loaded(self) -> bool:
        return False

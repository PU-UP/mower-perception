from __future__ import annotations

from typing import Any

from mowerseg.core.backend import InferenceBackend
from mowerseg.models.registry import ModelConfig


class HorizonBackend(InferenceBackend):
    """Skeleton for Horizon Robotics X5M / BPU runtime.

    Not implemented in this phase. Design boundary only:
    load compiled ``.bin`` artifact → infer tensors → return raw outputs.
    """

    name = "horizon"

    def __init__(self) -> None:
        self._loaded = False
        self._device_str = "bpu:unavailable"

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, model_config: ModelConfig) -> None:
        artifact = model_config.artifact_for("horizon")
        raise NotImplementedError(
            "HorizonBackend is a skeleton for the next stage (X5M / BPU). "
            f"Expected a compiled artifact such as model.bin "
            f"(configured={artifact!r} for model={model_config.name!r}). "
            "Do not call this backend until the Horizon SDK integration lands."
        )

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "HorizonBackend.infer is unavailable: Horizon SDK is not integrated yet."
        )

    def close(self) -> None:
        self._loaded = False

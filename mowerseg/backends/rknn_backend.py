from __future__ import annotations

from typing import Any

from mowerseg.core.backend import InferenceBackend
from mowerseg.models.registry import ModelConfig


class RKNNBackend(InferenceBackend):
    """Skeleton for Rockchip RK3588 / RKNPU runtime.

    Not implemented in this phase. Design boundary only:
    load compiled ``.rknn`` artifact → infer tensors → return raw outputs.
    """

    name = "rknn"

    def __init__(self) -> None:
        self._loaded = False
        self._device_str = "rknpu:unavailable"

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, model_config: ModelConfig) -> None:
        artifact = model_config.artifact_for("rknn")
        raise NotImplementedError(
            "RKNNBackend is a skeleton for a future RK3588 stage. "
            f"Expected a compiled artifact such as model.rknn "
            f"(configured={artifact!r} for model={model_config.name!r}). "
            "Do not call this backend until the RKNN Toolkit integration lands."
        )

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "RKNNBackend.infer is unavailable: RKNN Toolkit is not integrated yet."
        )

    def close(self) -> None:
        self._loaded = False

from __future__ import annotations

from typing import Any

import numpy as np

from mowerseg.core.backend import InferenceBackend
from mowerseg.models.registry import ModelConfig


class TorchBackend(InferenceBackend):
    """PyTorch / Hugging Face Transformers runtime."""

    name = "torch"

    def __init__(self) -> None:
        self._model: Any = None
        self._device_obj: Any = None
        self._device_str = "cpu"
        self._loader: str | None = None

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model_config: ModelConfig) -> None:
        import torch

        self._loader = model_config.loader
        self._device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device_str = str(self._device_obj)

        if model_config.loader == "transformers":
            from transformers import AutoModelForSemanticSegmentation

            source = model_config.artifact_for("torch") or model_config.hub_id
            if not source:
                raise ValueError(
                    f"Model '{model_config.name}' has no torch artifact or hub_id"
                )
            self._model = AutoModelForSemanticSegmentation.from_pretrained(source)
        else:
            raise NotImplementedError(
                f"TorchBackend loader '{model_config.loader}' is not implemented yet"
            )

        self._model.to(self._device_obj)
        self._model.eval()

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("TorchBackend.infer called before load()")

        import time

        import torch

        tensor_inputs = {
            key: self._as_torch(value).to(self._device_obj) for key, value in inputs.items()
        }

        if self._device_obj.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.inference_mode():
                outputs = self._model(**tensor_inputs)
            end.record()
            torch.cuda.synchronize()
            latency_ms = float(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            with torch.inference_mode():
                outputs = self._model(**tensor_inputs)
            latency_ms = (time.perf_counter() - t0) * 1000.0

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if not isinstance(logits, torch.Tensor):
            raise TypeError("TorchBackend expected tensor logits from the model")

        return {
            "logits": logits.detach().cpu().numpy(),
            "latency_ms": round(latency_ms, 1),
        }

    def close(self) -> None:
        self._model = None
        self._device_obj = None

    @staticmethod
    def _as_torch(value: Any) -> Any:
        import torch

        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        return torch.as_tensor(value)

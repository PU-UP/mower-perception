from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mowerseg.core.result import SemanticResult
from mowerseg.pipeline.engine import PerceptionEngine
from mowerseg.taxonomy import Taxonomy


@dataclass
class InferenceResult:
    """Backward-compatible CLI / FastAPI result shape."""

    image: Image.Image
    mower_mask: np.ndarray
    ade_mask: np.ndarray
    color_mask: Image.Image
    overlay: Image.Image
    stats: dict
    latency_ms: float
    model_name: str
    backend: str = "torch"
    task: str = "semantic_segmentation"
    device: str = "cpu"

    @classmethod
    def from_semantic(cls, result: SemanticResult) -> InferenceResult:
        if (
            result.image is None
            or result.class_mask is None
            or result.raw_mask is None
            or result.color_mask is None
            or result.overlay is None
        ):
            raise ValueError("SemanticResult is missing required visualization fields")

        hub_id = str(result.metadata.get("hub_id") or result.model_name)
        stats = result.to_stats()
        # Keep historical stats.model as HF hub id for the existing frontend.
        stats["model"] = hub_id
        stats["model_id"] = result.model_name
        return cls(
            image=result.image,
            mower_mask=result.class_mask,
            ade_mask=result.raw_mask,
            color_mask=result.color_mask,
            overlay=result.overlay,
            stats=stats,
            latency_ms=result.latency_ms,
            model_name=hub_id,
            backend=result.backend,
            task=result.task,
            device=result.device,
        )


class InferenceEngine:
    """Compatibility wrapper around PerceptionEngine."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self._engine = PerceptionEngine(config_path=self.config_path)
        self.taxonomy: Taxonomy = self._engine.taxonomy

    @property
    def backbone(self) -> Any:
        """Legacy attribute used by older health checks."""
        return self._engine

    @property
    def device(self) -> str:
        return self._engine.device

    def info(self) -> dict[str, Any]:
        return self._engine.info()

    def predict(self, image: Image.Image | str | Path) -> InferenceResult:
        result = self._engine.predict(image)
        if not isinstance(result, SemanticResult):
            raise TypeError(f"Expected SemanticResult, got {type(result)!r}")
        return InferenceResult.from_semantic(result)

    def save(self, result: InferenceResult, output_dir: str | Path, stem: str) -> dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "input": out / f"{stem}_input.png",
            "mask": out / f"{stem}_mask.png",
            "overlay": out / f"{stem}_overlay.png",
        }
        result.image.save(paths["input"])
        result.color_mask.save(paths["mask"])
        result.overlay.save(paths["overlay"])
        np.save(out / f"{stem}_label.npy", result.mower_mask)
        return {key: str(value) for key, value in paths.items()}

    def close(self) -> None:
        self._engine.close()

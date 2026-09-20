from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from mowerseg.core.frame import Frame
from mowerseg.models.registry import ModelConfig


def resize_long_side(image: Image.Image, long_side: int) -> Image.Image:
    width, height = image.size
    current = max(width, height)
    if current <= long_side:
        return image
    scale = long_side / current
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


class SegmentationPreprocessor:
    """Task-side preprocessing. Backend still only sees tensors."""

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self._processor: Any = None
        if model_config.loader == "transformers":
            from transformers import AutoImageProcessor

            source = model_config.artifact_for("torch") or model_config.hub_id
            if not source:
                raise ValueError(f"Model '{model_config.name}' missing hub_id/artifact")
            self._processor = AutoImageProcessor.from_pretrained(source)

    def __call__(self, frame: Frame) -> dict[str, Any]:
        rgb = frame.image.convert("RGB")
        resized = resize_long_side(rgb, self.model_config.input_long_side)

        if self._processor is not None:
            packed = self._processor(images=resized, return_tensors="np")
            return {key: np.asarray(value) for key, value in packed.items()}

        raise NotImplementedError(
            f"No preprocessor for loader={self.model_config.loader!r}. "
            "Configure a transformers model or extend SegmentationPreprocessor."
        )

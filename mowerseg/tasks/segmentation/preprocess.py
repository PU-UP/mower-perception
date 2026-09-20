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
        preprocess = self.model_config.preprocess or {}
        resize_mode = str(preprocess.get("resize_mode", "long_side"))

        if resize_mode == "long_side":
            image = resize_long_side(rgb, self.model_config.input_long_side)
        elif resize_mode == "processor":
            # Leave geometry entirely to the Hugging Face processor.
            image = rgb
        else:
            raise ValueError(
                f"Unsupported preprocess.resize_mode={resize_mode!r} "
                f"for model '{self.model_config.name}'"
            )

        if self._processor is not None:
            kwargs: dict[str, Any] = dict(preprocess.get("processor_kwargs") or {})
            if resize_mode == "long_side":
                # Keep full-frame coverage. Processors like MobileNetV2 default to
                # shortest-edge resize + center crop, which drops edges and then
                # gets incorrectly stretched back to the original canvas.
                kwargs.setdefault("do_resize", False)
                kwargs.setdefault("do_center_crop", False)
            packed = self._processor(images=image, return_tensors="np", **kwargs)
            return {key: np.asarray(value) for key, value in packed.items()}

        raise NotImplementedError(
            f"No preprocessor for loader={self.model_config.loader!r}. "
            "Configure a transformers model or extend SegmentationPreprocessor."
        )

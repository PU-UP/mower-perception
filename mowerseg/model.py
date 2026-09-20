"""Deprecated SegFormer helper.

Prefer::

    PerceptionEngine(task=..., model=..., backend=\"torch\")

This module remains for import compatibility only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from mowerseg.backends.torch_backend import TorchBackend
from mowerseg.core.frame import Frame
from mowerseg.models.registry import ModelConfig
from mowerseg.tasks.segmentation.postprocess import logits_to_label_mask
from mowerseg.tasks.segmentation.preprocess import SegmentationPreprocessor


@dataclass
class BackboneOutput:
    ade_mask: np.ndarray
    latency_ms: float


class SegFormerBackbone:
    """Legacy wrapper: SegFormer ADE20K inference via TorchBackend."""

    def __init__(self, model_name: str, long_side: int = 512) -> None:
        self.model_config = ModelConfig(
            name="segformer_b0_ade20k",
            task="semantic_segmentation",
            loader="transformers",
            hub_id=model_name,
            input_long_side=long_side,
            output_taxonomy="ade20k",
            requires_remapping=True,
        )
        self.backend = TorchBackend()
        self.backend.load(self.model_config)
        self._preprocessor = SegmentationPreprocessor(self.model_config)

    @property
    def device(self):
        return self.backend.device

    def predict(self, image: Image.Image) -> BackboneOutput:
        frame = Frame.from_image(image)
        inputs = self._preprocessor(frame)
        raw = self.backend.infer(inputs)
        width, height = frame.image.size
        ade_mask = logits_to_label_mask(raw["logits"], (height, width))
        return BackboneOutput(ade_mask=ade_mask, latency_ms=float(raw["latency_ms"]))

    def close(self) -> None:
        self.backend.close()

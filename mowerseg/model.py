from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


@dataclass
class BackboneOutput:
    ade_mask: np.ndarray
    latency_ms: float


class SegFormerBackbone:
    """Lightweight ADE20K segmentor used as the zero-shot student starter."""

    def __init__(self, model_name: str, long_side: int = 512) -> None:
        self.long_side = long_side
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _resize(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        long_side = max(width, height)
        if long_side <= self.long_side:
            return image
        scale = self.long_side / long_side
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(size, Image.Resampling.BILINEAR)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> BackboneOutput:
        rgb = image.convert("RGB")
        resized = self._resize(rgb)
        inputs = self.processor(images=resized, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            logits = self.model(**inputs).logits
            end.record()
            torch.cuda.synchronize()
            latency_ms = float(start.elapsed_time(end))
        else:
            import time

            t0 = time.perf_counter()
            logits = self.model(**inputs).logits
            latency_ms = (time.perf_counter() - t0) * 1000.0

        upsampled = torch.nn.functional.interpolate(
            logits,
            size=rgb.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        ade_mask = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return BackboneOutput(ade_mask=ade_mask, latency_ms=round(latency_ms, 1))

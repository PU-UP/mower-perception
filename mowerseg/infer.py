from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mowerseg.model import SegFormerBackbone
from mowerseg.taxonomy import Taxonomy, load_taxonomy
from mowerseg.visualize import colorize_mask, overlay_mask, summarize


@dataclass
class InferenceResult:
    image: Image.Image
    mower_mask: np.ndarray
    ade_mask: np.ndarray
    color_mask: Image.Image
    overlay: Image.Image
    stats: dict
    latency_ms: float
    model_name: str


class InferenceEngine:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.taxonomy: Taxonomy = load_taxonomy(self.config_path)
        self.backbone = SegFormerBackbone(
            self.taxonomy.model_name,
            long_side=self.taxonomy.input_long_side,
        )

    def predict(self, image: Image.Image | str | Path) -> InferenceResult:
        if isinstance(image, (str, Path)):
            rgb = Image.open(image).convert("RGB")
        else:
            rgb = image.convert("RGB")

        raw = self.backbone.predict(rgb)
        mower_mask = self.taxonomy.remap_ade(raw.ade_mask)
        color_mask = colorize_mask(mower_mask, self.taxonomy)
        overlay = overlay_mask(rgb, mower_mask, self.taxonomy)
        stats = summarize(mower_mask, self.taxonomy)
        stats["latency_ms"] = raw.latency_ms
        stats["model"] = self.taxonomy.model_name
        stats["device"] = str(self.backbone.device)
        return InferenceResult(
            image=rgb,
            mower_mask=mower_mask,
            ade_mask=raw.ade_mask,
            color_mask=color_mask,
            overlay=overlay,
            stats=stats,
            latency_ms=raw.latency_ms,
            model_name=self.taxonomy.model_name,
        )

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

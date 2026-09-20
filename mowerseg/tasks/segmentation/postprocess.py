from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from mowerseg.taxonomy import Taxonomy
from mowerseg.visualize import colorize_mask, overlay_mask, summarize


def logits_to_label_mask(logits: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Upsample logits to (height, width) and take argmax class ids."""

    if logits.ndim != 4:
        raise ValueError(f"Expected logits NCHW, got shape {logits.shape}")
    tensor = torch.from_numpy(logits.astype(np.float32))
    upsampled = torch.nn.functional.interpolate(
        tensor,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def apply_taxonomy_remap(
    raw_mask: np.ndarray,
    taxonomy: Taxonomy | None,
    *,
    requires_remapping: bool,
) -> np.ndarray:
    if not requires_remapping or taxonomy is None:
        return raw_mask
    return taxonomy.remap(raw_mask)


def build_semantic_views(
    image: Image.Image,
    class_mask: np.ndarray,
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    color_mask = colorize_mask(class_mask, taxonomy)
    overlay = overlay_mask(image, class_mask, taxonomy)
    class_stats = summarize(class_mask, taxonomy)
    return {
        "color_mask": color_mask,
        "overlay": overlay,
        "class_stats": class_stats,
    }

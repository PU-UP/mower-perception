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
    if logits.shape[0] != 1 or not 1 <= logits.shape[1] <= 256:
        raise ValueError("Expected one image and 1..256 classes")
    tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    # 4000x3000 annotations would allocate 7.2 GB for 150 upsampled channels.
    # Class chunks preserve exact resize-before-argmax semantics and first ties.
    best = torch.full(output_size, -float("inf"))
    labels = torch.zeros(output_size, dtype=torch.uint8)
    for start in range(0, tensor.shape[1], 8):
        enlarged = torch.nn.functional.interpolate(
            tensor[:, start:start+8], size=output_size,
            mode="bilinear", align_corners=False,
        )[0]
        scores, indices = enlarged.max(dim=0)
        update = scores > best
        best[update] = scores[update]
        labels[update] = (indices[update] + start).to(torch.uint8)
    return labels.numpy()


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

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from mowerseg.taxonomy import Taxonomy


def colorize_mask(mask: np.ndarray, taxonomy: Taxonomy) -> Image.Image:
    palette = taxonomy.palette
    color = palette[mask]
    return Image.fromarray(color, mode="RGB")


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    taxonomy: Taxonomy,
    alpha: float | None = None,
) -> Image.Image:
    strength = taxonomy.overlay_alpha if alpha is None else alpha
    color = colorize_mask(mask, taxonomy).convert("RGBA")
    color.putalpha(int(round(255 * strength)))
    base = image.convert("RGBA")
    return Image.alpha_composite(base, color).convert("RGB")


def encode_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def encode_jpeg(image: Image.Image, quality: int = 86) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def fit_long_side(image: Image.Image, long_side: int = 1280) -> Image.Image:
    width, height = image.size
    current = max(width, height)
    if current <= long_side:
        return image
    scale = long_side / current
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


def summarize(mask: np.ndarray, taxonomy: Taxonomy) -> dict:
    total = int(mask.size)
    rows: list[dict] = []
    traversable = 0
    safety = 0
    for item in taxonomy.classes:
        count = int((mask == item.id).sum())
        if count == 0:
            continue
        if item.traversable:
            traversable += count
        if item.safety:
            safety += count
        rows.append(
            {
                "id": item.id,
                "name": item.name,
                "name_zh": item.name_zh,
                "color": list(item.color),
                "pixels": count,
                "ratio": round(count / total, 4),
                "traversable": item.traversable,
                "safety": item.safety,
            }
        )
    rows.sort(key=lambda row: row["pixels"], reverse=True)
    return {
        "classes": rows,
        "traversable_ratio": round(traversable / total, 4),
        "safety_ratio": round(safety / total, 4),
        "pixels": total,
    }

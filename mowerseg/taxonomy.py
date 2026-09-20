from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Product-config keys that map a source dataset taxonomy onto mower classes.
SOURCE_TO_MOWER_KEYS = {
    "ade20k": "ade20k_to_mower",
    "pascal_voc": "pascal_voc_to_mower",
}


@dataclass(frozen=True)
class MowerClass:
    id: int
    name: str
    name_zh: str
    color: tuple[int, int, int]
    traversable: bool
    safety: bool


@dataclass(frozen=True)
class Taxonomy:
    """Mower product taxonomy (domain logic, not runtime/backend)."""

    classes: list[MowerClass]
    source_lookup: np.ndarray
    overlay_alpha: float
    input_long_side: int
    model_name: str
    samples: list[dict[str, str]]
    output_taxonomy: str = "mower"
    requires_remapping: bool = True

    @property
    def ade_lookup(self) -> np.ndarray:
        """Backward-compatible alias for source_lookup."""
        return self.source_lookup

    def by_id(self, class_id: int) -> MowerClass:
        return self.classes[class_id]

    @property
    def palette(self) -> np.ndarray:
        table = np.zeros((len(self.classes), 3), dtype=np.uint8)
        for item in self.classes:
            table[item.id] = item.color
        return table

    def remap(self, source_mask: np.ndarray) -> np.ndarray:
        return self.source_lookup[source_mask]

    def remap_ade(self, ade_mask: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for remap()."""
        return self.remap(ade_mask)


def load_product_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def build_source_lookup(
    classes: list[MowerClass],
    product_config: dict[str, Any],
    output_taxonomy: str,
) -> np.ndarray:
    """Build a 256-entry LUT from a source dataset taxonomy to mower class ids."""
    name_to_id = {item.name: item.id for item in classes}
    default = name_to_id.get("obstacle", 0)
    lookup = np.full(256, default, dtype=np.uint8)

    map_key = SOURCE_TO_MOWER_KEYS.get(output_taxonomy)
    mapping = product_config.get(map_key) if map_key else None
    if not mapping:
        return lookup

    for mower_name, source_ids in mapping.items():
        if mower_name not in name_to_id:
            continue
        target = name_to_id[mower_name]
        for source_id in source_ids or []:
            lookup[int(source_id)] = target
    return lookup


def taxonomy_from_product_config(
    raw: dict[str, Any],
    *,
    output_taxonomy: str | None = None,
    requires_remapping: bool | None = None,
    model_name: str | None = None,
) -> Taxonomy:
    classes = [
        MowerClass(
            id=int(item["id"]),
            name=str(item["name"]),
            name_zh=str(item["name_zh"]),
            color=tuple(int(c) for c in item["color"]),  # type: ignore[arg-type]
            traversable=bool(item.get("traversable", False)),
            safety=bool(item.get("safety", False)),
        )
        for item in raw["classes"]
    ]

    model_cfg: dict[str, Any] = raw.get("model", {})
    perception = raw.get("perception") or {}
    logical_model = str(
        model_name
        or perception.get("model")
        or model_cfg.get("name", "unknown")
    )
    source_taxonomy = str(output_taxonomy or "ade20k")
    lookup = build_source_lookup(classes, raw, source_taxonomy)

    if requires_remapping is None:
        map_key = SOURCE_TO_MOWER_KEYS.get(source_taxonomy)
        has_mapping = bool(map_key and raw.get(map_key))
        requires_remapping = has_mapping and bool(perception.get("requires_remapping", True))

    return Taxonomy(
        classes=classes,
        source_lookup=lookup,
        overlay_alpha=float(model_cfg.get("overlay_alpha", 0.45)),
        input_long_side=int(model_cfg.get("input_long_side", 512)),
        model_name=logical_model,
        samples=list(raw.get("samples", [])),
        output_taxonomy=source_taxonomy if requires_remapping else "mower",
        requires_remapping=bool(requires_remapping),
    )


def load_taxonomy(
    path: str | Path,
    *,
    output_taxonomy: str | None = None,
    requires_remapping: bool | None = None,
    model_name: str | None = None,
) -> Taxonomy:
    return taxonomy_from_product_config(
        load_product_config(path),
        output_taxonomy=output_taxonomy,
        requires_remapping=requires_remapping,
        model_name=model_name,
    )

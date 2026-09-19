from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


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
    classes: list[MowerClass]
    ade_lookup: np.ndarray
    overlay_alpha: float
    input_long_side: int
    model_name: str
    samples: list[dict[str, str]]

    def by_id(self, class_id: int) -> MowerClass:
        return self.classes[class_id]

    @property
    def palette(self) -> np.ndarray:
        table = np.zeros((len(self.classes), 3), dtype=np.uint8)
        for item in self.classes:
            table[item.id] = item.color
        return table

    def remap_ade(self, ade_mask: np.ndarray) -> np.ndarray:
        return self.ade_lookup[ade_mask]


def load_taxonomy(path: str | Path) -> Taxonomy:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
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
    name_to_id = {item.name: item.id for item in classes}
    lookup = np.full(256, name_to_id["obstacle"], dtype=np.uint8)
    for mower_name, ade_ids in raw.get("ade20k_to_mower", {}).items():
        target = name_to_id[mower_name]
        for ade_id in ade_ids or []:
            lookup[int(ade_id)] = target

    model_cfg: dict[str, Any] = raw.get("model", {})
    return Taxonomy(
        classes=classes,
        ade_lookup=lookup,
        overlay_alpha=float(model_cfg.get("overlay_alpha", 0.45)),
        input_long_side=int(model_cfg.get("input_long_side", 512)),
        model_name=str(model_cfg["name"]),
        samples=list(raw.get("samples", [])),
    )

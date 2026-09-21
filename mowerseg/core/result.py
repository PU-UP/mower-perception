from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class PerceptionResult:
    """Shared metadata for all perception task outputs."""

    frame_id: str | None = None
    timestamp: float | None = None
    latency_ms: float = 0.0
    model_name: str = ""
    backend: str = ""
    device: str = ""
    task: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_stats(self) -> dict[str, Any]:
        return {
            "input_shape": self.metadata.get("input_shape"),
            "latency_ms": self.latency_ms,
            "model": self.model_name,
            "backend": self.backend,
            "device": self.device,
            "task": self.task,
        }


@dataclass
class SemanticResult(PerceptionResult):
    """Canonical semantic-segmentation output for mower planning / visualization."""

    image: Image.Image | None = None
    class_mask: np.ndarray | None = None
    raw_mask: np.ndarray | None = None
    color_mask: Image.Image | None = None
    overlay: Image.Image | None = None
    class_stats: dict[str, Any] = field(default_factory=dict)

    def to_stats(self) -> dict[str, Any]:
        stats = dict(self.class_stats)
        stats.update(super().to_stats())
        return stats


@dataclass
class DepthResult(PerceptionResult):
    """Placeholder for future depth estimation outputs."""

    depth_map: np.ndarray | None = None
    confidence: np.ndarray | None = None
    min_depth: float | None = None
    max_depth: float | None = None


@dataclass
class DetectionResult(PerceptionResult):
    """Placeholder for future object detection outputs."""

    boxes: np.ndarray | None = None
    classes: np.ndarray | None = None
    scores: np.ndarray | None = None


@dataclass
class PrivacyResult(PerceptionResult):
    """Placeholder for future privacy / anonymization outputs."""

    anonymized_image: Image.Image | None = None
    regions: list[dict[str, Any]] = field(default_factory=list)

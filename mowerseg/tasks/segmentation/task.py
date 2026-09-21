from __future__ import annotations

from typing import Any

from mowerseg.core.backend import InferenceBackend
from mowerseg.core.frame import Frame
from mowerseg.core.result import SemanticResult
from mowerseg.core.task import PerceptionTask
from mowerseg.models.registry import ModelConfig
from mowerseg.tasks.segmentation.postprocess import (
    apply_taxonomy_remap,
    build_semantic_views,
    logits_to_label_mask,
)
from mowerseg.tasks.segmentation.preprocess import SegmentationPreprocessor
from mowerseg.taxonomy import Taxonomy, taxonomy_from_product_config


class SemanticSegmentationTask(PerceptionTask):
    """Semantic segmentation for mower traversability / safety perception."""

    name = "semantic_segmentation"

    def __init__(
        self,
        model_config: ModelConfig,
        backend: InferenceBackend,
        *,
        product_config: dict[str, Any] | None = None,
        taxonomy: Taxonomy | None = None,
        frequency_hz: float | None = None,
        enabled: bool = True,
        priority: int = 0,
    ) -> None:
        super().__init__(
            model_config,
            backend,
            product_config=product_config,
            frequency_hz=frequency_hz,
            enabled=enabled,
            priority=priority,
        )
        self.taxonomy = taxonomy
        self._preprocessor: SegmentationPreprocessor | None = None

    def setup(self) -> None:
        if self.taxonomy is None and self.product_config:
            self.taxonomy = taxonomy_from_product_config(self.product_config)
        if self.taxonomy is None:
            raise ValueError("SemanticSegmentationTask requires product taxonomy config")

        self.backend.load(self.model_config)
        self._preprocessor = SegmentationPreprocessor(self.model_config)

    def predict(self, frame: Frame) -> SemanticResult:
        if self._preprocessor is None or self.taxonomy is None:
            raise RuntimeError("SemanticSegmentationTask.setup() must be called first")

        inputs = self._preprocessor(frame)
        raw = self.backend.infer(inputs)
        logits = raw["logits"]
        latency_ms = float(raw.get("latency_ms", 0.0))

        width, height = frame.image.size
        raw_mask = logits_to_label_mask(logits, (height, width))
        class_mask = apply_taxonomy_remap(
            raw_mask,
            self.taxonomy,
            requires_remapping=self.model_config.requires_remapping,
        )
        views = build_semantic_views(frame.image, class_mask, self.taxonomy)

        result = SemanticResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp.timestamp() if frame.timestamp else None,
            latency_ms=latency_ms,
            model_name=self.model_config.name,
            backend=self.backend.name,
            device=self.backend.device,
            task=self.name,
            image=frame.image,
            class_mask=class_mask,
            raw_mask=raw_mask,
            color_mask=views["color_mask"],
            overlay=views["overlay"],
            class_stats=views["class_stats"],
            metadata={
                "input_shape": list(inputs["pixel_values"].shape),
                "preprocess": self.model_config.preprocess,
                "hub_id": self.model_config.hub_id,
                "output_taxonomy": self.model_config.output_taxonomy,
                "requires_remapping": self.model_config.requires_remapping,
            },
        )
        return result

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from mowerseg.backends.registry import create_backend
from mowerseg.core.frame import Frame
from mowerseg.core.result import PerceptionResult
from mowerseg.models.registry import ModelConfig, load_model_config
from mowerseg.tasks.registry import create_task
from mowerseg.taxonomy import Taxonomy, load_product_config, taxonomy_from_product_config


class PerceptionEngine:
    """Unified mower perception entry point: Task × Model × Backend."""

    def __init__(
        self,
        task: str = "semantic_segmentation",
        model: str = "segformer_b0_ade20k",
        backend: str = "torch",
        *,
        config_path: str | Path | None = None,
        product_config: dict[str, Any] | None = None,
        frequency_hz: float | None = None,
        enabled: bool = True,
        priority: int = 0,
    ) -> None:
        if product_config is None:
            if config_path is None:
                raise ValueError("PerceptionEngine requires config_path or product_config")
            product_config = load_product_config(config_path)

        perception = product_config.get("perception") or {}
        self.task_name = str(perception.get("task", task))
        self.model_name = str(perception.get("model", model))
        self.backend_name = str(perception.get("backend", backend))
        self.product_config = product_config
        self.config_path = Path(config_path) if config_path else None

        model_config = load_model_config(self.model_name)
        scheduling = (model_config.raw.get("scheduling") or {}) if model_config.raw else {}
        self.frequency_hz = (
            frequency_hz if frequency_hz is not None else scheduling.get("frequency_hz")
        )
        self.enabled = bool(scheduling.get("enabled", True)) if enabled is True else enabled
        self.priority = priority if priority != 0 else int(scheduling.get("priority", 0))

        self.model_config = self._merge_product_model_overrides(model_config, product_config)
        taxonomy = taxonomy_from_product_config(product_config)
        self.taxonomy: Taxonomy = replace(
            taxonomy,
            overlay_alpha=self.model_config.overlay_alpha,
            input_long_side=self.model_config.input_long_side,
            model_name=self.model_name,
            requires_remapping=self.model_config.requires_remapping,
        )

        self._backend = create_backend(self.backend_name)
        self._task = create_task(
            self.task_name,
            self.model_config,
            self._backend,
            product_config=product_config,
            taxonomy=self.taxonomy,
            frequency_hz=self.frequency_hz,
            enabled=self.enabled,
            priority=self.priority,
        )
        self._task.setup()

    @staticmethod
    def _merge_product_model_overrides(
        model_config: ModelConfig,
        product_config: dict[str, Any],
    ) -> ModelConfig:
        """Allow legacy configs/mower_seg.yaml model.* to override registry defaults."""
        legacy = product_config.get("model") or {}
        if not legacy:
            return model_config

        hub_id = model_config.hub_id
        if model_config.loader == "transformers" and legacy.get("name"):
            hub_id = str(legacy["name"])
        return replace(
            model_config,
            hub_id=hub_id,
            input_long_side=int(legacy.get("input_long_side", model_config.input_long_side)),
            overlay_alpha=float(legacy.get("overlay_alpha", model_config.overlay_alpha)),
        )

    @property
    def device(self) -> str:
        return self._backend.device

    def info(self) -> dict[str, Any]:
        return {
            "task": self.task_name,
            "model": self.model_name,
            "backend": self.backend_name,
            "device": self.device,
            "hub_id": self.model_config.hub_id,
            "frequency_hz": self.frequency_hz,
            "enabled": self.enabled,
            "priority": self.priority,
        }

    def predict(self, image: Any) -> PerceptionResult:
        if not self.enabled:
            raise RuntimeError("PerceptionEngine is disabled")
        frame = image if isinstance(image, Frame) else Frame.from_image(image)
        return self._task.predict(frame)

    def close(self) -> None:
        self._task.close()

    def __enter__(self) -> PerceptionEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def default_product_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "mower_seg.yaml"

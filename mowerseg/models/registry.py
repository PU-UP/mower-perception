from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class ModelConfig:
    """Logical model definition shared across backends."""

    name: str
    task: str
    display_name: str | None = None
    loader: str = "transformers"
    hub_id: str | None = None
    input_long_side: int = 512
    overlay_alpha: float = 0.46
    output_taxonomy: str = "ade20k"
    requires_remapping: bool = True
    num_classes: int | None = None
    artifacts: dict[str, str | None] = field(default_factory=dict)
    preprocess: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.display_name or self.name

    def artifact_for(self, backend: str) -> str | None:
        """Return backend-specific artifact path, if configured."""
        key = backend
        aliases = {
            "torch": ("torch", "pytorch"),
            "horizon": ("horizon", "horizon_x5", "x5m"),
            "rknn": ("rknn", "rk3588"),
        }
        for candidate in aliases.get(backend, (backend,)):
            value = self.artifacts.get(candidate)
            if value:
                return value
        if backend == "torch" and self.hub_id:
            return self.hub_id
        return None


def _parse_model_config(raw: dict[str, Any], *, name: str | None = None) -> ModelConfig:
    output = raw.get("output", {})
    input_cfg = raw.get("input", {})
    loader = str(raw.get("loader", raw.get("framework", "transformers")))
    hub_id = raw.get("hub_id")
    if hub_id is None and loader == "transformers":
        hub_id = raw.get("pretrained")
    resolved_name = str(name or raw.get("name"))
    display_name = raw.get("display_name")
    return ModelConfig(
        name=resolved_name,
        task=str(raw.get("task", "semantic_segmentation")),
        display_name=str(display_name) if display_name else None,
        loader=loader,
        hub_id=hub_id,
        input_long_side=int(input_cfg.get("long_side", raw.get("input_long_side", 512))),
        overlay_alpha=float(raw.get("overlay_alpha", 0.46)),
        output_taxonomy=str(output.get("taxonomy", "ade20k")),
        requires_remapping=bool(output.get("requires_remapping", True)),
        num_classes=output.get("num_classes"),
        artifacts=dict(raw.get("artifacts") or {}),
        preprocess=dict(raw.get("preprocess") or {}),
        raw=raw,
    )


def describe_model(name_or_path: str | Path) -> dict[str, Any]:
    """Return a JSON-serializable model card for UI / API listing."""
    cfg = load_model_config(name_or_path)
    return {
        "id": cfg.name,
        "display_name": cfg.label,
        "task": cfg.task,
        "loader": cfg.loader,
        "hub_id": cfg.hub_id,
        "output_taxonomy": cfg.output_taxonomy,
        "requires_remapping": cfg.requires_remapping,
        "num_classes": cfg.num_classes,
        "input_long_side": cfg.input_long_side,
    }


def list_model_cards() -> list[dict[str, Any]]:
    return [describe_model(name) for name in list_models()]


def load_model_config(name_or_path: str | Path) -> ModelConfig:
    """Load a model config by registry name or yaml path."""
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return _parse_model_config(raw, name=raw.get("name") or path.stem)

    name = str(name_or_path)
    candidate = CONFIG_DIR / f"{name}.yaml"
    if not candidate.exists():
        raise KeyError(f"Unknown model config: {name} (expected {candidate})")
    raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    return _parse_model_config(raw, name=name)


def list_models() -> list[str]:
    return sorted(path.stem for path in CONFIG_DIR.glob("*.yaml"))

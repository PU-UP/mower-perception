from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mowerseg.backends.registry import create_backend, list_backends
from mowerseg.models.registry import list_models, load_model_config
from mowerseg.tasks.registry import list_tasks
from mowerseg.tasks.segmentation.postprocess import apply_taxonomy_remap, logits_to_label_mask
from mowerseg.taxonomy import load_taxonomy, taxonomy_from_product_config


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_CFG = ROOT / "configs" / "mower_seg.yaml"


def test_list_backends_includes_torch_and_skeletons():
    names = list_backends()
    assert names == ["horizon", "rknn", "torch"]


def test_create_unsupported_backend_raises():
    with pytest.raises(KeyError, match="Unsupported backend"):
        create_backend("tensorrt_fantasy")


def test_horizon_and_rknn_are_explicit_skeletons():
    model = load_model_config("segformer_b0_ade20k")
    horizon = create_backend("horizon")
    rknn = create_backend("rknn")
    with pytest.raises(NotImplementedError, match="HorizonBackend"):
        horizon.load(model)
    with pytest.raises(NotImplementedError, match="RKNNBackend"):
        rknn.load(model)


def test_model_config_loading_and_artifacts():
    cfg = load_model_config("segformer_b0_ade20k")
    assert cfg.task == "semantic_segmentation"
    assert cfg.loader == "transformers"
    assert cfg.hub_id == "nvidia/segformer-b0-finetuned-ade-512-512"
    assert cfg.output_taxonomy == "ade20k"
    assert cfg.requires_remapping is True
    assert cfg.artifact_for("torch") == cfg.hub_id
    assert "segformer_b0_ade20k" in list_models()


def test_model_config_custom_artifact_path(tmp_path: Path):
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        """
name: custom_mower
task: semantic_segmentation
loader: transformers
hub_id: example/model
output:
  taxonomy: mower
  requires_remapping: false
artifacts:
  torch: /weights/torch/model.safetensors
  horizon_x5: /weights/horizon_x5/model.bin
  rk3588: /weights/rk3588/model.rknn
""",
        encoding="utf-8",
    )
    cfg = load_model_config(yaml_path)
    assert cfg.requires_remapping is False
    assert cfg.artifact_for("torch") == "/weights/torch/model.safetensors"
    assert cfg.artifact_for("horizon") == "/weights/horizon_x5/model.bin"
    assert cfg.artifact_for("rknn") == "/weights/rk3588/model.rknn"


def test_taxonomy_remapping():
    taxonomy = load_taxonomy(PRODUCT_CFG)
    ade = np.zeros((4, 4), dtype=np.uint8)
    ade[0, 0] = 9   # grass
    ade[0, 1] = 2   # sky
    ade[0, 2] = 12  # person
    ade[0, 3] = 99  # unlisted -> obstacle
    mower = taxonomy.remap_ade(ade)
    assert mower[0, 0] == 1
    assert mower[0, 1] == 9
    assert mower[0, 2] == 5
    assert mower[0, 3] == 7


def test_apply_taxonomy_remap_can_skip():
    taxonomy = load_taxonomy(PRODUCT_CFG)
    raw = np.arange(4, dtype=np.uint8).reshape(2, 2)
    kept = apply_taxonomy_remap(raw, taxonomy, requires_remapping=False)
    assert np.array_equal(kept, raw)
    remapped = apply_taxonomy_remap(raw, taxonomy, requires_remapping=True)
    assert remapped.shape == raw.shape
    assert not np.array_equal(remapped, raw)


def test_logits_to_label_mask_shape():
    logits = np.zeros((1, 5, 8, 10), dtype=np.float32)
    logits[0, 3, :, :] = 10.0
    mask = logits_to_label_mask(logits, (16, 20))
    assert mask.shape == (16, 20)
    assert mask.dtype == np.uint8
    assert int(mask.max()) == 3


def test_product_config_perception_section():
    raw = {
        "perception": {
            "task": "semantic_segmentation",
            "model": "segformer_b0_ade20k",
            "backend": "torch",
        },
        "model": {"name": "nvidia/segformer-b0-finetuned-ade-512-512", "overlay_alpha": 0.5},
        "classes": [
            {
                "id": 0,
                "name": "ignore",
                "name_zh": "忽略",
                "color": [0, 0, 0],
                "traversable": False,
                "safety": False,
            },
            {
                "id": 1,
                "name": "grass",
                "name_zh": "可割草坪",
                "color": [0, 255, 0],
                "traversable": True,
                "safety": False,
            },
            {
                "id": 7,
                "name": "obstacle",
                "name_zh": "障碍",
                "color": [1, 2, 3],
                "traversable": False,
                "safety": False,
            },
        ],
        "ade20k_to_mower": {"grass": [9], "obstacle": [0]},
    }
    taxonomy = taxonomy_from_product_config(raw)
    assert taxonomy.model_name == "segformer_b0_ade20k"
    assert taxonomy.by_id(1).traversable is True


def test_list_tasks():
    assert "semantic_segmentation" in list_tasks()


def test_semantic_result_stats_shape():
    from mowerseg.core.result import SemanticResult

    result = SemanticResult(
        latency_ms=12.5,
        model_name="segformer_b0_ade20k",
        backend="torch",
        device="cpu",
        task="semantic_segmentation",
        class_stats={"classes": [], "pixels": 0},
    )
    stats = result.to_stats()
    assert stats["backend"] == "torch"
    assert stats["task"] == "semantic_segmentation"
    assert stats["latency_ms"] == 12.5

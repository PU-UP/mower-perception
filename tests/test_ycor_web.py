from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mowerseg.taxonomy import load_taxonomy
from mowerseg.models.registry import describe_model

ROOT = Path(__file__).resolve().parents[1]


def test_proxy_taxonomy_preserves_binary_without_safety_claims():
    taxonomy = load_taxonomy(ROOT / "configs/mower_seg.yaml", output_taxonomy="ycor_proxy")
    assert [(c.id, c.name) for c in taxonomy.classes] == [(0, "other_valid"), (1, "traversable_grass")]
    assert all(not c.traversable and not c.safety for c in taxonomy.classes)
    assert not taxonomy.requires_remapping
    assert describe_model("lraspp_ycor")["num_classes"] == 2


def test_missing_web_checkpoint_has_actionable_error(monkeypatch, tmp_path):
    from fastapi import HTTPException
    from mowerseg import server
    monkeypatch.setattr(server, "_engines", {})
    monkeypatch.setenv("YCOR_CHECKPOINT", str(tmp_path / "missing.pt"))
    with pytest.raises(HTTPException) as error:
        server.get_engine("lraspp_ycor")
    assert error.value.status_code == 503
    assert "YCOR_CHECKPOINT" in error.value.detail


def test_real_web_pipeline_matches_offline_proxy():
    import os
    import torch
    from mowerseg.infer import InferenceEngine
    from mowerseg.ycor import ProxyPredictor
    checkpoint = Path(os.environ.get("YCOR_CHECKPOINT", ROOT / "weights/ycor/best.pt"))
    if not checkpoint.exists():
        pytest.skip("Local trained artifact not installed")
    torch.set_num_threads(4)
    engine = InferenceEngine(ROOT / "configs/mower_seg.yaml", model="lraspp_ycor")
    image = Image.open(ROOT / "assets/samples/lawn_path.jpg").convert("RGB")
    result = engine.predict(image)
    predictor = ProxyPredictor("lraspp", checkpoint, engine.device)
    expected, _ = predictor.predict(image)
    assert np.array_equal(result.ade_mask, expected)
    assert np.array_equal(result.mower_mask, expected)
    assert result.image.size == result.overlay.size == result.color_mask.size == (512, 384)
    assert set(np.unique(result.mower_mask)) <= {0, 1}
    assert result.stats["traversable_ratio"] == result.stats["safety_ratio"] == 0
    assert sum(c["pixels"] for c in result.stats["classes"]) == 512 * 384
    engine.close()

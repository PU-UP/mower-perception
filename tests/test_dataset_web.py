import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from mowerseg import server
from mowerseg.models.hf_cache import from_pretrained_cached


def test_catalog_only_lists_complete_pairs_and_rejects_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation/grassseghb-256.json").write_text(json.dumps({"samples": [
        {"image": "images/0G1.jpg", "mask": "annotations/0G1_mask.png", "garden": "0"}]}))
    assert server.grass_catalog() == {"total": 1, "samples": []}
    for file in ("images/0G1.jpg", "annotations/0G1_mask.png"):
        path = tmp_path / "data/grassseghb" / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert server.grass_catalog()["samples"] == [{"id": "0G1", "garden": "0"}]
    with pytest.raises(HTTPException) as exc:
        server.grass_sample("../../etc/passwd")
    assert exc.value.status_code == 404


def test_stale_report_is_not_shown(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    for file, value in [("evaluation/grassseghb-256.json", {}),
                        ("outputs/grassseghb/results.json", {"complete": True, "manifest_sha256": "stale"})]:
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    assert server.grass_report() is None


def test_cache_first_and_missing_cache_network_fallback():
    class Factory:
        calls = []
        missing = False

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            cls.calls.append(kwargs)
            if cls.missing and kwargs.get("local_files_only"):
                raise OSError("not cached")
            return source

    assert from_pretrained_cached(Factory, "model", revision="fixed") == "model"
    assert Factory.calls == [{"local_files_only": True, "revision": "fixed"}]
    Factory.calls.clear()
    Factory.missing = True
    from_pretrained_cached(Factory, "model", revision="fixed")
    assert Factory.calls == [{"local_files_only": True, "revision": "fixed"}, {"revision": "fixed"}]


def test_summary_uses_aggregate_metrics_not_one_image(monkeypatch):
    aggregate = {"mowable_iou": 0.8}
    report = {"sample_count": 256, "code_revision": "fixed", "hardware": {}, "protocol": {},
              "models": {"model": {"config": {"display_name": "Model"}, "metrics": aggregate,
                                   "model_latency": {"median_ms": 10},
                                   "end_to_end_latency": {"median_ms": 20},
                                   "samples": [{"garden": "1", "input_shape": [1, 3, 600, 800], "original_size": [4000, 3000], "metrics": {"mowable_iou": 0.1}}]}}}
    monkeypatch.setattr(server, "grass_report", lambda: report)
    monkeypatch.setattr(server, "grass_manifest", lambda: {"samples": [{"garden": "1"}, {"garden": "2"}]})
    result = server.grass_summary()
    assert result["sample_count"] == 256 and result["garden_count"] == 1
    assert result["models"][0]["metrics"] == aggregate
    assert "samples" not in result["models"][0]
    monkeypatch.setattr(server, "grass_report", lambda: None)
    assert server.grass_summary() == {"available": False}


@pytest.mark.parametrize("fault", ["partial", "missing", "duplicate", "foreign", "garden", "hash", "none"])
def test_report_requires_each_models_full_manifest(tmp_path, monkeypatch, fault):
    import hashlib
    import copy
    monkeypatch.setattr(server, "ROOT", tmp_path)
    samples = [{"image": f"images/{i}.jpg", "mask": f"masks/{i}.png", "garden": str(i),
                "image_sha256": "image-hash", "mask_sha256": "mask-hash"} for i in range(2)]
    manifest = tmp_path / "evaluation/grassseghb-256.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"samples": samples}))
    report = {"complete": True, "sample_count": 2, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "models": {"a": {"samples": copy.deepcopy(samples)}, "b": {"samples": copy.deepcopy(samples)}}}
    rows = report["models"]["b"]["samples"]
    if fault == "partial":
        report["sample_count"] = 1
        for model in report["models"].values():
            model["samples"] = model["samples"][:1]
    elif fault == "missing": rows.pop()
    elif fault == "duplicate": rows[1] = rows[0]
    elif fault == "foreign": rows[1]["image"] = "elsewhere.jpg"
    elif fault == "garden": rows[1]["garden"] = "wrong"
    elif fault == "hash": rows[1]["mask_sha256"] = "changed"
    path = tmp_path / "outputs/grassseghb/results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))
    assert (server.grass_report() is not None) == (fault == "none")


@pytest.mark.parametrize("radius", [0, 10])
def test_nondefault_boundary_protocol_reaches_both_views(monkeypatch, radius):
    import numpy as np
    from PIL import Image
    import mowerseg.evaluate
    sample = {"image": "images/example.jpg", "garden": "7"}
    row = {**sample, "metrics": {}, "input_shape": [1, 3, 240, 320], "original_size": [640, 480]}
    report = {"sample_count": 1, "code_revision": "test", "hardware": {},
              "protocol": {"boundary_radius_pixels": radius}, "models": {
                  "segformer_b0_ade20k": {"config": {"display_name": "B0"}, "metrics": {},
                                         "model_latency": {}, "end_to_end_latency": {}, "samples": [row]}}}
    monkeypatch.setattr(server, "grass_report", lambda: report)
    monkeypatch.setattr(server, "grass_sample", lambda _: sample)
    monkeypatch.setattr(mowerseg.evaluate, "load_pair", lambda *_: (Image.new("RGB", (4, 3)), np.zeros((3, 4), dtype=np.uint8)))
    summary = server.grass_summary()
    detail = server.grass_detail("example")
    assert summary["protocol"]["boundary_radius_pixels"] == radius
    assert detail["protocol"]["boundary_radius_pixels"] == radius
    assert summary["garden_count"] == 1
    assert summary["models"][0]["input_shapes"] == [(240, 320)]
    assert summary["models"][0]["original_sizes"] == [(640, 480)]


def test_smoke_run_cannot_overwrite_full_report(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mowerseg.evaluate import run
    monkeypatch.chdir(tmp_path)
    default = tmp_path / "outputs/grassseghb"
    default.mkdir(parents=True)
    saved = default / "results.json"
    original = '{"sample_count": 256}'
    saved.write_text(original)
    with pytest.raises(ValueError, match="separate --output"):
        run(SimpleNamespace(limit=1, output=default))
    assert saved.read_text() == original
    other = tmp_path / "other"
    other.mkdir()
    (other / "results.json").write_text(original)
    with pytest.raises(ValueError, match="larger report"):
        run(SimpleNamespace(limit=1, output=other))
    assert (other / "results.json").read_text() == original

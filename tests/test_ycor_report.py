import json
import shutil
from pathlib import Path

import pytest
from mowerseg.ycor_report import summary, demo_image, demo_samples

DIRECTORY = Path(__file__).resolve().parents[1] / "evaluation/ycor"


def test_frozen_report_has_complete_models_and_stages():
    result = summary()
    assert result["available"] and result["sample_count"] == 145
    assert {m["id"] for m in result["models"]} == {"lraspp", "b0"}
    assert all(len(m["per_group"]) == 5 for m in result["models"])
    assert all(m["latency"]["pipeline_ms"]["n"] == 120 for m in result["models"])


@pytest.mark.parametrize("damage", ["coverage", "digest", "counts", "timing"])
def test_rejects_incomplete_or_modified_report(tmp_path, damage):
    for name in ("results.json", "split.json"):
        shutil.copyfile(DIRECTORY / name, tmp_path / name)
    p = tmp_path / "results.json"
    report = json.loads(p.read_text())
    model = report["models"]["lraspp"]
    if damage == "coverage": model["samples"].pop()
    if damage == "digest": model["samples"][0]["image_sha256"] = "bad"
    if damage == "counts": model["counts"]["tp"] += 1
    if damage == "timing": model["latency"]["pipeline_ms"]["p50_ms"] = 0
    p.write_text(json.dumps(report))
    with pytest.raises(ValueError): summary(tmp_path)


def test_demo_samples_are_whitelisted_and_real_when_installed():
    with pytest.raises(FileNotFoundError): demo_image("../../etc/passwd")
    for sample in demo_samples():
        image = demo_image(sample["id"])
        assert image.mode == "RGB" and image.size == (1024, 544)

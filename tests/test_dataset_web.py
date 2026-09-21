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

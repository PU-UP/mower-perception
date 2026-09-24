from __future__ import annotations

import base64
import hashlib
import json
import numpy as np
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

from mowerseg.infer import InferenceEngine
from mowerseg.ycor_report import summary as ycor_summary_report, demo_samples, demo_image
from mowerseg.models.registry import list_model_cards, list_models
from mowerseg.taxonomy import load_product_config, load_taxonomy
from mowerseg.visualize import encode_jpeg, encode_png, fit_long_side

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "mower_seg.yaml"
SAMPLES = ROOT / "assets" / "samples"

_engines: dict[str, InferenceEngine] = {}


def _default_model_id() -> str:
    product = load_product_config(CONFIG)
    perception = product.get("perception") or {}
    return str(perception.get("model") or "segformer_b0_ade20k")


def get_engine(model: str | None = None) -> InferenceEngine:
    """Lazy-load and cache an InferenceEngine per logical model id."""
    model_id = model or _default_model_id()
    if model_id not in list_models():
        raise HTTPException(status_code=404, detail=f"未知模型: {model_id}")
    engine = _engines.get(model_id)
    if engine is None:
        try:
            engine = InferenceEngine(CONFIG, model=model_id)
        except (OSError, ValueError) as exc:
            if model_id == "lraspp_ycor":
                raise HTTPException(status_code=503, detail="YCOR 权重缺失或无效；请配置 YCOR_CHECKPOINT 指向已训练的 best.pt。") from exc
            raise
        _engines[model_id] = engine
    return engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_engine()
    yield
    for engine in list(_engines.values()):
        engine.close()
    _engines.clear()


app = FastAPI(title="MowerSeg", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if SAMPLES.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")


def _to_data_url(image, fmt: str = "png") -> str:
    payload = encode_jpeg(image) if fmt == "jpeg" else encode_png(image)
    mime = "image/jpeg" if fmt == "jpeg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


def _run(image: Image.Image, *, model: str | None = None, truth=None) -> dict:
    engine = get_engine(model)
    result = engine.predict(image)
    preview_input = fit_long_side(result.image)
    preview_mask = fit_long_side(result.color_mask)
    preview_overlay = fit_long_side(result.overlay)
    info = engine.info()
    evaluation = None
    if truth is not None and info.get("output_taxonomy") == "ade20k":
        from mowerseg.evaluate import counts, metrics
        evaluation = metrics(counts(result.ade_mask == 9, truth))
    return {
        "raw_mask": (_to_data_url(Image.fromarray(result.ade_mask.astype(np.uint8)))
                     if info.get("output_taxonomy") == "ycor_proxy" else None),
        "evaluation": evaluation,
        "evaluation_protocol": {"boundary_radius_pixels": 3} if evaluation is not None else None,
        "overlay": _to_data_url(preview_overlay, "jpeg"),
        "mask": _to_data_url(preview_mask, "png"),
        "input": _to_data_url(preview_input, "jpeg"),
        "stats": result.stats,
        "model": {
            "id": info["model"],
            "display_name": info.get("display_name") or info["model"],
            "hub_id": info.get("hub_id"),
            "backend": info["backend"],
            "output_taxonomy": info.get("output_taxonomy"),
        },
        "taxonomy": [
            {
                "id": item.id,
                "name": item.name,
                "name_zh": item.name_zh,
                "color": list(item.color),
                "traversable": item.traversable,
                "safety": item.safety,
            }
            for item in engine.taxonomy.classes
            if item.name != "ignore"
        ],
    }


@app.get("/api/health")
def health() -> dict:
    ready = True
    try:
        info = get_engine().info()
        device = info["device"]
        model = info["model"]
        backend = info["backend"]
        task = info["task"]
    except Exception as exc:  # pragma: no cover - startup diagnostics
        ready = False
        device = "unknown"
        model = str(exc)
        backend = "unknown"
        task = "unknown"
    return {
        "ok": ready,
        "device": device,
        "model": model,
        "backend": backend,
        "task": task,
        "models": list_models(),
    }


@app.get("/api/models")
def models() -> dict:
    default = _default_model_id()
    return {
        "default": default,
        "models": list_model_cards(),
    }


@app.get("/api/taxonomy")
def taxonomy(model: str | None = Query(default=None)) -> dict:
    engine = get_engine(model)
    info = engine.info()
    cfg = load_taxonomy(
        CONFIG,
        output_taxonomy=info.get("output_taxonomy"),
        requires_remapping=engine.taxonomy.requires_remapping,
        model_name=info["model"],
    )
    return {
        "model": info.get("hub_id") or cfg.model_name,
        "model_id": info["model"],
        "display_name": info.get("display_name") or info["model"],
        "backend": info["backend"],
        "task": info["task"],
        "device": info["device"],
        "output_taxonomy": info.get("output_taxonomy"),
        "default_model": _default_model_id(),
        "models": list_model_cards(),
        "classes": [
            {
                "id": item.id,
                "name": item.name,
                "name_zh": item.name_zh,
                "color": list(item.color),
                "traversable": item.traversable,
                "safety": item.safety,
            }
            for item in cfg.classes
        ],
        "samples": cfg.samples + demo_samples(),
    }


@app.post("/api/infer")
async def infer(
    file: UploadFile = File(...),
    model: str | None = Query(default=None),
) -> dict:
    try:
        payload = await file.read()
        image = Image.open(BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法读取图片: {exc}") from exc
    return _run(image, model=model)


@app.post("/api/infer-sample/{sample_id}")
def infer_sample(
    sample_id: str,
    model: str | None = Query(default=None),
) -> dict:
    if sample_id.startswith("ycor-demo-"):
        try:
            image = demo_image(sample_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _run(image, model=model)
    if sample_id.startswith("grass-"):
        from mowerseg.evaluate import load_pair
        sample = grass_sample(sample_id.removeprefix("grass-"))
        try:
            image, truth = load_pair(ROOT / "data/grassseghb", sample)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=f"数据缺失或校验失败: {exc}") from exc
        return _run(image, model=model, truth=truth)
    taxonomy = load_taxonomy(CONFIG)
    match = next((item for item in taxonomy.samples if item["id"] == sample_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="样例不存在")
    path = SAMPLES / match["file"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="样例文件缺失")
    return _run(Image.open(path).convert("RGB"), model=model)


def grass_manifest():
    path = ROOT / "evaluation/grassseghb-256.json"
    return json.loads(path.read_text()) if path.exists() else {"samples": []}


def grass_sample(sample_id):
    sample = next((s for s in grass_manifest()["samples"]
                   if Path(s["image"]).stem == sample_id), None)
    if sample is None:
        raise HTTPException(status_code=404, detail="未知评测样本")
    return sample


def grass_report():
    path = ROOT / "outputs/grassseghb/results.json"
    manifest = ROOT / "evaluation/grassseghb-256.json"
    if not path.exists() or not manifest.exists():
        return None
    report = json.loads(path.read_text())
    if not report.get("complete") or report.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
        return None
    expected = {s["image"]: s for s in json.loads(manifest.read_text())["samples"]}
    if not expected or report.get("sample_count") != len(expected) or not report.get("models"):
        return None
    for model in report["models"].values():
        rows = model.get("samples", [])
        if len(rows) != len(expected) or {r["image"] for r in rows} != set(expected):
            return None
        if any(any(r.get(key) != expected[r["image"]].get(key)
                   for key in ("garden", "mask", "image_sha256", "mask_sha256")) for r in rows):
            return None
    return report


@app.get("/api/grass")
def grass_catalog():
    samples = grass_manifest()["samples"]
    available = [s for s in samples if all((ROOT / "data/grassseghb" / s[k]).is_file()
                                         for k in ("image", "mask"))]
    return {"total": len(samples), "samples": [
        {"id": Path(s["image"]).stem, "garden": s["garden"]} for s in available]}


@app.get("/api/grass/{sample_id}")
def grass_detail(sample_id: str):
    from mowerseg.evaluate import load_pair
    sample = grass_sample(sample_id)
    try:
        image, truth = load_pair(ROOT / "data/grassseghb", sample)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=f"数据缺失或校验失败: {exc}") from exc
    label = Image.fromarray(truth.astype(np.uint8) * 255)
    preview = fit_long_side(image, 800)
    label = label.resize(preview.size, Image.Resampling.NEAREST)
    report = grass_report()
    comparisons = []
    for model, values in (report or {}).get("models", {}).items():
        row = next((r for r in values["samples"] if r["image"] == sample["image"]), None)
        if row and model in list_models():
            comparisons.append({"model": model, "name": values["config"]["display_name"],
                                "metrics": row["metrics"],
                                "image": f"/api/grass/{sample_id}/comparison/{model}"})
    return {"id": sample_id, "input": _to_data_url(preview, "jpeg"),
            "truth": _to_data_url(label), "comparisons": comparisons,
            "revision": (report or {}).get("code_revision"),
            "protocol": (report or {}).get("protocol")}


@app.get("/api/grass/{sample_id}/comparison/{model}")
def grass_comparison(sample_id: str, model: str):
    grass_sample(sample_id)
    report = grass_report()
    if not report or model not in report["models"] or model not in list_models():
        raise HTTPException(status_code=404, detail="没有对应离线评测")
    path = ROOT / "outputs/grassseghb" / model / f"{sample_id}-comparison.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="本地对比图尚未生成")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/grass-summary")
def grass_summary():
    report = grass_report()
    if report is None:
        return {"available": False}
    return {
        "available": True, "sample_count": report["sample_count"],
        "garden_count": len({s["garden"] for m in report["models"].values() for s in m["samples"]}),
        "revision": report["code_revision"], "hardware": report["hardware"],
        "protocol": report["protocol"],
        "models": [{"id": key, "name": value["config"]["display_name"],
                    "metrics": value["metrics"], "model_latency": value["model_latency"],
                    "end_to_end_latency": value["end_to_end_latency"],
                    "input_shapes": sorted({tuple(s["input_shape"][2:]) for s in value["samples"]}),
                    "original_sizes": sorted({tuple(s["original_size"]) for s in value["samples"]})}
                   for key, value in report["models"].items()],
    }


@app.get("/api/ycor-summary")
def ycor_summary():
    try:
        return ycor_summary_report()
    except (OSError, ValueError, KeyError, TypeError):
        return {"available": False, "reason": "YCOR 完整报告缺失或校验失败，请恢复 evaluation/ycor 中的冻结报告。"}

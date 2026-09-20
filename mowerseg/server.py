from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from mowerseg.infer import InferenceEngine
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
        engine = InferenceEngine(CONFIG, model=model_id)
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


def _run(image: Image.Image, *, model: str | None = None) -> dict:
    engine = get_engine(model)
    result = engine.predict(image)
    preview_input = fit_long_side(result.image)
    preview_mask = fit_long_side(result.color_mask)
    preview_overlay = fit_long_side(result.overlay)
    info = engine.info()
    return {
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
        "samples": cfg.samples,
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
    taxonomy = load_taxonomy(CONFIG)
    match = next((item for item in taxonomy.samples if item["id"] == sample_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="样例不存在")
    path = SAMPLES / match["file"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="样例文件缺失")
    return _run(Image.open(path).convert("RGB"), model=model)

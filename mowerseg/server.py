from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from mowerseg.infer import InferenceEngine
from mowerseg.taxonomy import load_taxonomy
from mowerseg.visualize import encode_jpeg, encode_png, fit_long_side

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "mower_seg.yaml"
SAMPLES = ROOT / "assets" / "samples"


@lru_cache(maxsize=1)
def engine() -> InferenceEngine:
    return InferenceEngine(CONFIG)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine()
    yield


app = FastAPI(title="MowerSeg", version="0.1.0", lifespan=lifespan)
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


def _run(image: Image.Image) -> dict:
    result = engine().predict(image)
    preview_input = fit_long_side(result.image)
    preview_mask = fit_long_side(result.color_mask)
    preview_overlay = fit_long_side(result.overlay)
    return {
        "overlay": _to_data_url(preview_overlay, "jpeg"),
        "mask": _to_data_url(preview_mask, "png"),
        "input": _to_data_url(preview_input, "jpeg"),
        "stats": result.stats,
        "taxonomy": [
            {
                "id": item.id,
                "name": item.name,
                "name_zh": item.name_zh,
                "color": list(item.color),
                "traversable": item.traversable,
                "safety": item.safety,
            }
            for item in engine().taxonomy.classes
            if item.name != "ignore"
        ],
    }


@app.get("/api/health")
def health() -> dict:
    ready = True
    try:
        device = str(engine().backbone.device)
        model = engine().taxonomy.model_name
    except Exception as exc:  # pragma: no cover - startup diagnostics
        ready = False
        device = "unknown"
        model = str(exc)
    return {"ok": ready, "device": device, "model": model}


@app.get("/api/taxonomy")
def taxonomy() -> dict:
    cfg = load_taxonomy(CONFIG)
    return {
        "model": cfg.model_name,
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
async def infer(file: UploadFile = File(...)) -> dict:
    try:
        payload = await file.read()
        image = Image.open(BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法读取图片: {exc}") from exc
    return _run(image)


@app.post("/api/infer-sample/{sample_id}")
def infer_sample(sample_id: str) -> dict:
    taxonomy = load_taxonomy(CONFIG)
    match = next((item for item in taxonomy.samples if item["id"] == sample_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="样例不存在")
    path = SAMPLES / match["file"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="样例文件缺失")
    return _run(Image.open(path).convert("RGB"))

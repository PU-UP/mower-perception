"""Reproducible grass-as-mowable proxy evaluation on labelled binary masks."""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import subprocess
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from mowerseg.pipeline.engine import PerceptionEngine


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def boundary(mask):
    """Both sides of four-neighbour transitions; image perimeter is not an edge."""
    edge = np.zeros(mask.shape, dtype=bool)
    dy = mask[1:] != mask[:-1]
    dx = mask[:, 1:] != mask[:, :-1]
    edge[1:] |= dy
    edge[:-1] |= dy
    edge[:, 1:] |= dx
    edge[:, :-1] |= dx
    return edge


def dilate(mask, radius):
    # Separable square (Chebyshev) neighbourhood, without a scipy dependency.
    horizontal = mask.copy()
    for offset in range(1, radius+1):
        horizontal[:, offset:] |= mask[:, :-offset]
        horizontal[:, :-offset] |= mask[:, offset:]
    result = horizontal.copy()
    for offset in range(1, radius+1):
        result[offset:] |= horizontal[:-offset]
        result[:-offset] |= horizontal[offset:]
    return result


def counts(pred, truth, radius=3):
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("Prediction/annotation geometry mismatch")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("GrassSegHB must contain only 0/1 labels; no inferred ignore value")
    if not np.isin(pred, [0, 1]).all():
        raise ValueError("Predictions must be binary")
    pred, truth = pred.astype(bool), truth.astype(bool)
    pb, tb = boundary(pred), boundary(truth)
    return {
        "tp": int(np.count_nonzero(pred & truth)),
        "fp": int(np.count_nonzero(pred & ~truth)),
        "tn": int(np.count_nonzero(~pred & ~truth)),
        "fn": int(np.count_nonzero(~pred & truth)),
        "pred_boundary": int(pb.sum()), "true_boundary": int(tb.sum()),
        "matched_pred_boundary": int(np.count_nonzero(pb & dilate(tb, radius))),
        "matched_true_boundary": int(np.count_nonzero(tb & dilate(pb, radius))),
        "boundary_band_pixels": int(dilate(tb, radius).sum()),
        "boundary_band_errors": int(np.count_nonzero((pred != truth) & dilate(tb, radius))),
    }


def metrics(c):
    precision = ratio(c["matched_pred_boundary"], c["pred_boundary"])
    recall = ratio(c["matched_true_boundary"], c["true_boundary"])
    f1 = None if precision is None or recall is None else (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    return {
        "mowable_false_negative_rate": ratio(c["fn"], c["tp"] + c["fn"]),
        "mowable_iou": ratio(c["tp"], c["tp"] + c["fp"] + c["fn"]),
        "nonmowable_false_positive_rate": ratio(c["fp"], c["fp"] + c["tn"]),
        "predicted_mowable_error_fraction": ratio(c["fp"], c["tp"] + c["fp"]),
        "boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1,
        "boundary_band_error_rate": ratio(c["boundary_band_errors"], c["boundary_band_pixels"]),
    }


def summarize_times(values):
    return {"count": len(values), "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def load_pair(root, sample):
    # Check file identities before timing; never silently substitute missing data.
    for key in ("image", "mask"):
        path = (root / sample[key]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Manifest path escapes dataset root")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if sample.get(key + "_sha256") != digest:
            raise ValueError(f"Missing/mismatched manifest hash: {path}")
    image = Image.open(root / sample["image"]).convert("RGB")
    truth = np.asarray(Image.open(root / sample["mask"]))
    if truth.ndim != 2 or image.size != truth.shape[::-1] or not np.isin(truth, [0, 1]).all():
        raise ValueError(f"Invalid annotation geometry/values: {sample['mask']}")
    return image, truth


def save_comparison(path, image, truth, pred, title):
    size = (480, round(480 * image.height / image.width))
    canvas = Image.new("RGB", (size[0]*4, size[1]+40), "white")
    error = np.zeros((*pred.shape, 3), dtype=np.uint8)
    error[pred & (truth == 0)] = (255, 60, 60)
    error[~pred & (truth == 1)] = (50, 100, 255)
    panels = [image, Image.fromarray(truth.astype(np.uint8)*255),
              Image.fromarray(pred.astype(np.uint8)*255), Image.fromarray(error)]
    for i, panel in enumerate(panels):
        canvas.paste(panel.resize(size, Image.Resampling.NEAREST if i else Image.Resampling.BILINEAR), (i*size[0], 40))
    ImageDraw.Draw(canvas).text((5, 5), title + " | RGB | GT | prediction | red=FP blue=FN", fill="black")
    canvas.save(path)


def run(args):
    if args.limit and args.output.resolve() == Path("outputs/grassseghb").resolve():
        raise ValueError("--limit requires a separate --output (for example outputs/smoke); refusing to overwrite the full report")
    existing = args.output / "results.json"
    if args.limit and existing.exists():
        previous = json.loads(existing.read_text())
        if previous.get("sample_count", 0) > args.limit:
            raise ValueError("Smoke run would overwrite a larger report; choose a new --output")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not manifest.get("download_complete") or not manifest.get("samples"):
        raise ValueError("Complete the labelled subset download first")
    samples = manifest["samples"]
    if args.limit:
        samples = samples[:args.limit]
    if len({s["image"] for s in samples}) != len(samples):
        raise ValueError("Duplicate images in manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    report = {
        "software": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "transformers", "numpy", "pillow")},
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "sample_count": len(samples), "seed": manifest["seed"],
        "scope": "exploratory grass proxy, not full mower safety evaluation",
        "hardware": {"platform": platform.platform(), "cpu": platform.processor(),
                     "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                     "torch": torch.__version__, "cuda": torch.version.cuda, "threads": args.threads,
                     "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else platform.processor(),
                     "matmul_precision": torch.get_float32_matmul_precision(),
                     "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                     "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
        "protocol": {"warmup": args.warmup, "batch_size": 1, "precision": "float32",
                     "boundary_radius_pixels": args.boundary_radius,
                     "boundary": "4-neighbour transition pixels, square Chebyshev tolerance at original annotation resolution; no image perimeter",
                     "aggregation": "micro: sum confusion/boundary counts across images; undefined ratios are null",
                     "model_ms": "forward only; CUDA events + synchronization; excludes transfers and output resizing",
                     "end_to_end_ms": "warm-cache file decode + preprocess + transfers + forward + full-resolution postprocess + product views; excludes model load, hash verification, metric calculation and artifact saving",
                     "order": args.models},
        "models": {},
    }
    for name in args.models:
        rows = []
        with PerceptionEngine(model=name, config_path="configs/mower_seg.yaml") as engine:
            if engine.model_config.output_taxonomy != "ade20k":
                raise ValueError("This comparison requires ADE20K models with grass index 9")
            image, _ = load_pair(args.root, samples[0])
            for _ in range(args.warmup):
                warm = engine.predict(image)
                del warm
            directory = args.output / name
            directory.mkdir(exist_ok=True)
            for index, sample in enumerate(samples):
                _, truth = load_pair(args.root, sample)
                synchronize(engine.device)
                t0 = time.perf_counter()
                result = engine.predict(args.root / sample["image"])
                synchronize(engine.device)
                elapsed = (time.perf_counter() - t0)*1000
                pred = result.raw_mask == 9
                if not np.array_equal(pred, result.class_mask == 1):
                    raise ValueError("Product grass mapping differs from ADE20K grass-only proxy")
                c = counts(pred, truth, args.boundary_radius)
                row = {**sample, "counts": c, "metrics": metrics(c),
                       "model_ms": result.latency_ms, "end_to_end_ms": elapsed,
                       "input_shape": result.metadata["input_shape"],
                       "original_size": list(result.image.size)}
                rows.append(row)
                stem = Path(sample["image"]).stem
                Image.fromarray(pred.astype(np.uint8)).save(directory / f"{stem}.png")
                # Contact sheets for all samples permit unbiased visual inspection.
                save_comparison(directory / f"{stem}-comparison.jpg", result.image, truth, pred, f"{stem} {name}")
                if (index+1) % 8 == 0:
                    print(f"{name}: {index+1}/{len(samples)}", flush=True)
                del result
            total = {k: sum(r["counts"][k] for r in rows) for k in rows[0]["counts"]}
            worst = sorted(rows, key=lambda r: r["metrics"]["nonmowable_false_positive_rate"] or 0, reverse=True)[:10]
            per_garden = {}
            for garden in sorted({r["garden"] for r in rows}, key=int):
                members = [r for r in rows if r["garden"] == garden]
                subtotal = {k: sum(r["counts"][k] for r in members) for k in total}
                per_garden[garden] = {"n": len(members), "metrics": metrics(subtotal)}
            report["models"][name] = {
                "config": engine.model_config.raw, "device": engine.device,
                "counts": total, "metrics": metrics(total), "per_garden": per_garden,
                "model_latency": summarize_times([r["model_ms"] for r in rows]),
                "end_to_end_latency": summarize_times([r["end_to_end_ms"] for r in rows]),
                "worst_false_positive_samples": [r["image"] for r in worst], "samples": rows,
            }
        report["complete"] = len(report["models"]) == len(args.models)
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    write_gallery(args.output, report)
    print(json.dumps({k: {"metrics": v["metrics"], "model_latency": v["model_latency"], "end_to_end_latency": v["end_to_end_latency"]} for k, v in report["models"].items()}, indent=2))


def write_gallery(output, report):
    parts = ["<!doctype html><meta charset='utf-8'><title>GrassSegHB comparison</title>",
             "<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:16px}img{width:100%}pre{white-space:pre-wrap}</style>",
             "<h1>GrassSegHB: grass-only proxy comparison</h1>",
             "<p>Exploratory evaluation, not mower safety certification. White = mowable; red = false positive, blue = false negative. No raw images are included in Git.</p>"]
    for name, model in report["models"].items():
        parts.append(f"<h2>{html.escape(name)}</h2><pre>{html.escape(json.dumps(model['metrics'], indent=2))}</pre>")
    failures = dict.fromkeys(s for m in report["models"].values() for s in m["worst_false_positive_samples"])
    parts.append("<h2>Worst false-positive cases, same image across models</h2>")
    for source in failures:
        stem = Path(source).stem
        parts.append(f"<h3>{html.escape(stem)}</h3>")
        for name in report["models"]:
            path = f"{name}/{stem}-comparison.jpg"
            parts.append(f"<p>{html.escape(name)}</p><img loading='lazy' src='{html.escape(path)}' alt='RGB, annotation, prediction and errors'>")
    (output/"index.html").write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/grassseghb"))
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/grassseghb-256.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/grassseghb"))
    parser.add_argument("--models", nargs="+", default=["segformer_b0_ade20k", "mit_resnet18_ade20k"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Smoke run only; report records reduced count")
    args = parser.parse_args()
    if args.warmup < 1 or args.threads < 1 or args.boundary_radius < 0 or args.limit < 0:
        parser.error("Invalid warmup/threads/radius/limit")
    run(args)


if __name__ == "__main__":
    main()

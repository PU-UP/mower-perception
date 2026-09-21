"""Data-free LR-ASPP compatibility probe, NOT training or a speed benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import torch
import torchvision
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Never overwrite a prior probe or an evaluation artifact.
    if args.output.exists():
        parser.error("Output already exists; use a new filename")
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA; no silent CPU fallback")
    torch.set_num_threads(4)
    torch.manual_seed(20260921)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    # Official factory strictly loads ImageNet classification weights, then
    # retains the dilated feature extractor. The ENTIRE segmentation head is random.
    model = lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=weights, num_classes=2
    ).cuda().train()
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[1]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if not digest.startswith("8738ca79"):
        raise ValueError("Unexpected official ImageNet checkpoint hash")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    # Diagnostic tensor only: no images, labels, loss supervision or optimizer.
    # This is NOT a few-sample overfit check and yields no mowable capability.
    diagnostic = torch.randn(2, 3, 384, 512, device="cuda")
    out = model(diagnostic)["out"]
    assert out.shape == (2, 2, 384, 512) and torch.isfinite(out).all()
    out.square().mean().backward()
    probes = {
        "backbone": model.backbone["0"][0].weight.grad,
        "low_classifier": model.classifier.low_classifier.weight.grad,
        "high_classifier": model.classifier.high_classifier.weight.grad,
    }
    for name, grad in probes.items():
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
            raise RuntimeError(f"Missing/nonfinite/zero gradient: {name}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    source = Path(inspect.getfile(inspect.unwrap(lraspp_mobilenet_v3_large)))
    report = {
        "status": "compatibility_probe_passed_training_NOT_started",
        "utc": datetime.now(timezone.utc).isoformat(),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_steps": 0, "trained_checkpoint": None,
        "scope": "random diagnostic tensor; no data, labels, optimizer, accuracy or latency benchmark",
        "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                     "torchvision_commit": torchvision.version.git_version},
        "hardware": {"platform": platform.platform(), "gpu": torch.cuda.get_device_name(),
                     "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                     "free_disk_bytes": shutil.disk_usage(Path.cwd()).free,
                     "threads": torch.get_num_threads(),
                     "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()},
        "model": {"architecture": "torchvision LR-ASPP MobileNetV3-Large, dilated backbone",
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "weights_backbone": str(weights), "weights": None,
                  "head": "randomly initialized, untrained; no mowable predictions available",
                  "pretrained_url": weights.url, "pretrained_local_path": str(checkpoint),
                  "pretrained_sha256": digest,
                  "official_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "probe": {"input_nchw": list(diagnostic.shape), "output_nchw": list(out.shape),
                  "precision": "float32", "tf32": False,
                  "finite_nonzero_gradients": list(probes),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "one_cold_forward_backward_seconds_NOT_inference_latency": elapsed},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

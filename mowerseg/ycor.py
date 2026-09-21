"""YCOR traversable-grass proxy: not a mowability or obstacle classifier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import torch
from torch.nn import functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from mowerseg.evaluate import boundary, dilate, ratio
from mowerseg.models.hf_cache import from_pretrained_cached

TITLE = "YCOR 可通行草地代理模型"
SIZE = (512, 384)  # width, height; shared output requirement
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]
B0_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
B0_REV = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(image):
    a = np.asarray(image, dtype=np.float32).copy().transpose(2, 0, 1) / 255
    return (torch.from_numpy(a) - MEAN) / STD


def proxy_counts(pred, truth, radius=2):
    """Ignore 255; exclude an (r+1) band around void from both boundary sets."""
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("Prediction/annotation geometry mismatch")
    if not np.isin(truth, [0, 1, 255]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("Expected binary predictions and 0/1/255 truth")
    if radius < 0:
        raise ValueError("Negative boundary radius")
    valid = truth != 255
    p, t = pred.astype(bool), truth == 1
    safe = ~dilate(~valid, radius + 1)
    pb, tb = boundary(p) & safe, boundary(t) & safe
    return {"tp": int((p & t & valid).sum()), "fp": int((p & ~t & valid).sum()),
            "tn": int((~p & ~t & valid).sum()), "fn": int((~p & t & valid).sum()),
            "ignored": int((~valid).sum()), "pred_boundary": int(pb.sum()),
            "true_boundary": int(tb.sum()),
            "matched_pred_boundary": int((pb & dilate(tb, radius)).sum()),
            "matched_true_boundary": int((tb & dilate(pb, radius)).sum())}


def proxy_metrics(c):
    p = ratio(c['matched_pred_boundary'], c['pred_boundary'])
    r = ratio(c['matched_true_boundary'], c['true_boundary'])
    return {"grass_iou": ratio(c['tp'], c['tp']+c['fp']+c['fn']),
            "grass_false_positive_rate": ratio(c['fp'], c['fp']+c['tn']),
            "grass_false_negative_rate": ratio(c['fn'], c['tp']+c['fn']),
            "predicted_grass_error_fraction": ratio(c['fp'], c['tp']+c['fp']),
            "boundary_precision": p, "boundary_recall": r,
            "boundary_f1": None if p is None or r is None else (2*p*r/(p+r) if p+r else 0.)}


def sum_counts(rows):
    return {k: sum(row[k] for row in rows) for k in rows[0]}


def load_checkpoint(path, device='cuda'):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('task') != 'ycor_traversable_grass_proxy' or data.get('optimizer_steps', 0) < 1:
        raise ValueError('Not a trained YCOR proxy checkpoint')
    if data.get('labels') != {'0': 'other_valid', '1': 'traversable_grass', '255': 'ignore'}:
        raise ValueError('Checkpoint label mapping mismatch')
    if data.get('output_size_wh') != list(SIZE):
        raise ValueError('Unsupported output geometry')
    model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=2)
    model.load_state_dict(data['model'], strict=True)
    return model.to(device).eval(), data


class ProxyPredictor:
    """Same CPU uint8 512x384 output; timing includes H2D and D2H transfers."""
    def __init__(self, name, checkpoint=None, device='cuda'):
        self.name, self.device = name, torch.device(device)
        self.processor = None
        if name == 'lraspp':
            if checkpoint is None:
                raise ValueError('LR-ASPP requires a trained checkpoint')
            self.model, self.metadata = load_checkpoint(checkpoint, device)
        elif name == 'b0':
            from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
            self.model = from_pretrained_cached(AutoModelForSemanticSegmentation, B0_ID, revision=B0_REV).to(device).eval()
            self.processor = from_pretrained_cached(AutoImageProcessor, B0_ID, revision=B0_REV)
            if self.model.config.num_labels != 150 or self.model.config.id2label[9] != 'grass':
                raise ValueError('Unexpected ADE20K taxonomy')
        else:
            raise ValueError('Unknown proxy model')

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, decoded_rgb):
        # Decoding is excluded: API requires an already-decoded RGB PIL image.
        if decoded_rgb.mode != 'RGB':
            raise ValueError('Pass a decoded RGB image')
        self.sync()
        start = time.perf_counter()
        if self.processor is None:
            x = normalize(decoded_rgb.resize(SIZE, Image.Resampling.BILINEAR))[None]
        else:
            x = self.processor(images=decoded_rgb, return_tensors='pt')['pixel_values']
        x = x.to(self.device)  # H2D is preprocessing, blocking transfer
        self.sync()
        prep = time.perf_counter()
        logits = self.model(x)['out'] if self.name == 'lraspp' else self.model(pixel_values=x).logits
        self.sync()
        forward = time.perf_counter()
        logits = F.interpolate(logits, size=SIZE[::-1], mode='bilinear', align_corners=False)
        labels = logits.argmax(1)[0]
        pred = (labels == (1 if self.name == 'lraspp' else 9)).to(torch.uint8).cpu().numpy()
        self.sync()  # D2H and argmax are postprocessing
        end = time.perf_counter()
        return pred, {"preprocess_ms": (prep-start)*1000,
                      "forward_ms": (forward-prep)*1000,
                      "postprocess_ms": (end-forward)*1000,
                      "pipeline_ms": (end-start)*1000,
                      "input_nchw": list(x.shape)}


def main():
    parser = argparse.ArgumentParser(description=TITLE + '；0=其他有效类别，1=可通行草地；非安全可割判断')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    if args.output.suffix.lower() != '.png':
        parser.error('Binary label masks must be saved as lossless PNG')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = ProxyPredictor('lraspp', args.checkpoint, args.device)
    with Image.open(args.image) as source:
        rgb = source.convert('RGB')
    pred, timing = model.predict(rgb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_info = PngInfo()
    png_info.add_text('Model', TITLE)
    png_info.add_text('Labels', '0=other valid semantic classes; 1=traversable grass proxy; NOT verified mowability')
    Image.fromarray(pred).save(args.output, pnginfo=png_info)
    print(json.dumps({'model': TITLE, 'checkpoint_sha256': sha256(args.checkpoint),
                      'labels': {'0': '其他有效类别（不代表具体障碍类型）', '1': '可通行草地代理'},
                      'output_wh': list(SIZE), 'timing': timing}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""One bounded LR-ASPP training run on a frozen YCOR proxy split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from mowerseg.ycor import SIZE, MEAN, STD, sha256
from tools.prepare_ycor import read_binary, checked_samples


def load_arrays(root, samples):
    images, labels = [], []
    for s in samples:
        with Image.open(root/s['image']) as image:
            images.append(np.asarray(image.convert('RGB').resize(SIZE, Image.Resampling.BILINEAR)))
        labels.append(read_binary(root/s['mask'], SIZE))
    return np.stack(images), np.stack(labels)


def batch(images, labels, indices, augment):
    x = images[indices].copy()
    y = labels[indices].copy()
    if augment:
        for i in range(len(x)):
            if random.random() < .5:
                x[i], y[i] = x[i, :, ::-1].copy(), y[i, :, ::-1].copy()
            gain = random.uniform(.8, 1.2)
            x[i] = np.clip(x[i].astype(np.float32)*gain, 0, 255).astype(np.uint8)
    x = torch.from_numpy(x.transpose(0, 3, 1, 2)).float()/255
    return ((x-MEAN)/STD).cuda(), torch.from_numpy(y.astype(np.int64)).cuda()


@torch.inference_mode()
def validate(model, images, labels, batch_size):
    model.eval()
    confusion = torch.zeros(4, dtype=torch.int64, device='cuda')
    loss_sum, valid_sum = 0., 0
    for start in range(0, len(images), batch_size):
        x, y = batch(images, labels, np.arange(start, min(start+batch_size, len(images))), False)
        out = model(x)['out']
        loss_sum += float(F.cross_entropy(out, y, ignore_index=255, reduction='sum'))
        valid = y != 255
        valid_sum += int(valid.sum())
        confusion += torch.bincount((y[valid]*2+out.argmax(1)[valid]).flatten(), minlength=4)
    tn, fp, fn, tp = confusion.tolist()
    return {'loss': loss_sum/valid_sum, 'iou': tp/(tp+fp+fn) if tp+fp+fn else None,
            'counts': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}}


def make_model():
    return lraspp_mobilenet_v3_large(weights=None, weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
                                    num_classes=2).cuda()


def curve(logs, path):
    width, height = 800, 360
    def points(key, scale):
        return ' '.join(f'{45+i*710/max(1,len(logs)-1):.1f},{320-min(1,r[key]/scale)*270:.1f}' for i,r in enumerate(logs))
    max_loss = max(r['train_loss'] for r in logs)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
                   '<rect width="100%" height="100%" fill="white"/>'
                   '<text x="30" y="25">YCOR proxy training: blue=validation IoU [0,1]; red=train CE (scaled)</text>'
                   '<path d="M45 50V320H760" stroke="black" fill="none"/>'
                   f'<polyline points="{points("val_iou",1)}" fill="none" stroke="#1468c4" stroke-width="2"/>'
                   f'<polyline points="{points("train_loss",max_loss)}" fill="none" stroke="#ca4235" stroke-width="2"/>'
                   f'<text x="45" y="345">epoch 1 to {len(logs)}; CE scale 0 to {max_loss:.4f}</text></svg>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/ycor/yamaha_v0'))
    parser.add_argument('--manifest', type=Path, default=Path('evaluation/ycor/split.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--overfit', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or args.epochs < 1 or args.batch_size < 2:
        parser.error('Use a new output directory, positive epochs and batch >=2')
    args.output.mkdir(parents=True)
    seed = 20260921
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set by the launch command.
    manifest = json.loads(args.manifest.read_text())
    train = checked_samples(args.root, manifest, 'train')
    if args.overfit:
        chosen = [s for s in train if .1 < s['positive_fraction'] < .85][:4]
        if len(chosen) != 4:
            raise ValueError('Need four nontrivial real training examples')
        train = chosen
    images, labels = load_arrays(args.root, train)
    if not args.overfit:
        val_images, val_labels = load_arrays(args.root, checked_samples(args.root, manifest, 'val'))
    model = make_model()
    lr = .003 if args.overfit else .001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
    config = {'task': 'ycor_traversable_grass_proxy', 'seed': seed, 'epochs': None if args.overfit else args.epochs,
              'batch_size': 4 if args.overfit else args.batch_size, 'lr': lr, 'optimizer': 'AdamW', 'weight_decay': .0001,
              'loss': 'unweighted CrossEntropy, ignore_index=255', 'precision': 'FP32, TF32 off',
              'input_wh': list(SIZE), 'output_wh': list(SIZE), 'augmentation': 'none' if args.overfit else 'horizontal flip p=.5; brightness gain uniform .8..1.2; no crop',
              'resize': 'whole image PIL bilinear; label nearest', 'normalization': 'RGB/255 ImageNet mean/std',
              'pretrained': 'MobileNet_V3_Large_Weights.IMAGENET1K_V1; entire head random',
              'manifest_sha256': sha256(args.manifest), 'train_count': len(train),
              'validation_count': 0 if args.overfit else len(val_images), 'overfit_only': args.overfit,
              'torch': str(torch.__version__), 'device': torch.cuda.get_device_name(),
              'selection': 'overfit diagnostic only; no checkpoint selection' if args.overfit else 'highest validation aggregate positive IoU; argmax; no test access',
              'budget': 'at most 300 optimizer updates' if args.overfit else f'{args.epochs} epochs, at most 45 minutes; no hyperparameter search',
              'threads': 4, 'deterministic_algorithms': 'enabled with warn_only; CUDA interpolation may not be bitwise reproducible'}
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    started = time.perf_counter()
    if args.overfit:
        logs = []
        for step in range(1, 301):
            model.train()
            x, y = batch(images, labels, np.arange(4), False)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x)['out'], y, ignore_index=255)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); optimizer.step()
            if step % 20 == 0:
                score = validate(model, images, labels, 4)
                logs.append({'step': step, 'train_loss': float(loss.detach()), **score})
                print(json.dumps(logs[-1]), flush=True)
                if score['iou'] >= .93 and score['loss'] < .15: break
        result = {'sample_ids': [s['id'] for s in train], 'optimizer_steps': step,
                  'elapsed_seconds': time.perf_counter()-started, 'logs': logs,
                  'passed': score['iou'] >= .9 and score['loss'] < .2}
        (args.output/'overfit.json').write_text(json.dumps(result, indent=2)+'\n')
        if not result['passed']: raise RuntimeError('Real-sample overfit check did not pass')
        return
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.00001)
    logs, steps, best = [], 0, -1.
    for epoch in range(1, args.epochs+1):
        epoch_started = time.perf_counter()
        model.train()
        order = np.random.permutation(len(images))
        loss_sum, pixels = 0., 0
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start+args.batch_size]
            x, y = batch(images, labels, ids, True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)['out']
            loss = F.cross_entropy(out, y, ignore_index=255)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); optimizer.step(); steps += 1
            n = int((y != 255).sum())
            loss_sum += float(loss.detach())*n; pixels += n
        score = validate(model, val_images, val_labels, args.batch_size)
        row = {'epoch': epoch, 'optimizer_steps': steps, 'train_loss': loss_sum/pixels,
               'val_loss': score['loss'], 'val_iou': score['iou'], 'val_counts': score['counts'],
               'lr': optimizer.param_groups[0]['lr'], 'seconds': time.perf_counter()-epoch_started}
        logs.append(row)
        print(json.dumps(row), flush=True)
        if score['iou'] > best:
            best = score['iou']
            torch.save({'task': config['task'], 'model': model.state_dict(), 'epoch': epoch,
                        'optimizer_steps': steps, 'labels': {'0': 'other_valid', '1': 'traversable_grass', '255': 'ignore'},
                        'output_size_wh': list(SIZE), 'manifest_sha256': config['manifest_sha256'],
                        'config': config, 'validation': score}, args.output/'best.pt')
        scheduler.step()
        (args.output/'history.json').write_text(json.dumps(logs, indent=2)+'\n')
        curve(logs, args.output/'learning-curve.svg')
        if time.perf_counter()-started > 45*60: break
    (args.output/'completed.json').write_text(json.dumps({'epochs_completed': len(logs),
        'optimizer_steps': steps, 'elapsed_seconds': time.perf_counter()-started,
        'best_val_iou': best, 'best_checkpoint_sha256': sha256(args.output/'best.pt'),
        'checkpoint_path': str((args.output/'best.pt').resolve()), 'test_evaluated': False}, indent=2)+'\n')


if __name__ == '__main__':
    main()

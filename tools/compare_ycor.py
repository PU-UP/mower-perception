"""Frozen-test accuracy and decoded-RGB offline latency, at the same 512x384 output."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from PIL import Image, ImageDraw
import torch

from mowerseg.ycor import ProxyPredictor, SIZE, TITLE, B0_ID, B0_REV, sha256, proxy_counts, proxy_metrics, sum_counts
from tools.prepare_ycor import checked_samples, read_binary


def summarize(values):
    return {'n': len(values), 'p50_ms': float(np.percentile(values, 50)),
            'p95_ms': float(np.percentile(values, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/ycor/yamaha_v0'))
    parser.add_argument('--manifest', type=Path, default=Path('evaluation/ycor/split.json'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new output directory; never overwrite a report')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads(args.manifest.read_text())
    samples = checked_samples(args.root, manifest, 'test')
    models = {name: ProxyPredictor(name, args.checkpoint) for name in ('lraspp','b0')}
    if models['lraspp'].metadata['manifest_sha256'] != sha256(args.manifest):
        raise ValueError('Checkpoint was trained against a different split')
    # Freeze evidence before any final-test inference. No tuning parameters in this tool.
    frozen = {'checkpoint_sha256': sha256(args.checkpoint), 'manifest_sha256': sha256(args.manifest),
              'comparison_code_sha256': sha256(__file__),
              'b0_source': {'hub_id': B0_ID, 'revision': B0_REV},
              'output_wh': list(SIZE), 'classification': 'argmax after bilinear logits interpolation; align_corners=False',
              'boundary_radius': 2, 'selected_epoch': models['lraspp'].metadata['epoch'],
              'utc': datetime.now(timezone.utc).isoformat()}
    (args.output/'frozen.json').write_text(json.dumps(frozen, indent=2)+'\n')
    # Multi-image benchmark selection uses only frozen sample IDs, never performance.
    benchmark_ids = np.linspace(0, len(samples)-1, min(12,len(samples)), dtype=int).tolist()
    decoded = []
    for i in benchmark_ids:
        with Image.open(args.root/samples[i]['image']) as image:
            decoded.append(image.convert('RGB'))
    for model in models.values():
        for i in range(30): model.predict(decoded[i % len(decoded)])
    timings = {name: [] for name in models}
    for repeat in range(10):
        for i, image in enumerate(decoded):
            order = ('lraspp','b0') if (repeat+i)%2 == 0 else ('b0','lraspp')
            for name in order:
                _, timing = models[name].predict(image)
                timings[name].append({'sample_id': samples[benchmark_ids[i]]['id'], 'repeat': repeat, **timing})
    report = {'task': TITLE, 'scope': '方案比较：领域训练 LR-ASPP vs 未领域训练 B0 grass 代理；非安全可割评测',
              'frozen': frozen, 'complete': False, 'sample_count': len(samples),
              'software': {n: importlib.metadata.version(n) for n in ['torch','torchvision','transformers','numpy','pillow']},
              'hardware': {'gpu': torch.cuda.get_device_name(), 'platform': platform.platform(),
                           'cpu': next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
                           'cuda': torch.version.cuda, 'threads': 4, 'precision': 'FP32', 'tf32': False},
              'timing_protocol': {'batch': 1, 'warmup_per_model': 30, 'images': len(decoded), 'repeats_per_image': 10,
                 'order': 'alternate model order for each repeat/image',
                 'preprocess': 'decoded RGB PIL -> resize/normalize -> CPU tensor -> H2D; CUDA synchronize',
                 'forward': 'model forward; CUDA synchronize',
                 'postprocess': 'logits bilinear to 384x512 -> argmax -> binary -> D2H numpy uint8; CUDA synchronize',
                 'pipeline': 'wall time encompassing all three stages including transfers and synchronization',
                 'excluded': 'disk I/O, image decoding (predecoded RGB input), load, download, visualization, save, metrics, HTTP',
                 'limitations': 'laptop power/temperature not locked; both models resident; local machine only'},
              'boundary': '4-neighbour transitions, both sides, no frame perimeter; radius 2 Chebyshev at 512x384; exclude all boundary pixels within radius+1 of ignore in BOTH masks',
              'models': {}}
    for name, model in models.items():
        rows = []
        mask_directory = args.output / name
        mask_directory.mkdir()
        for i, s in enumerate(samples):
            with Image.open(args.root/s['image']) as image:
                rgb = image.convert('RGB')
            truth = read_binary(args.root/s['mask'], SIZE)
            pred, _ = model.predict(rgb)
            c = proxy_counts(pred, truth)
            prediction_path = mask_directory / (s['id'] + '.png')
            Image.fromarray(pred).save(prediction_path)  # outside the timed benchmark
            rows.append({'id': s['id'], 'group': s['report_group'], 'split_group': s['group'], 'image_sha256': s['image_sha256'],
                         'prediction_sha256': sha256(prediction_path),
                         'mask_sha256': s['mask_sha256'], 'counts': c, 'metrics': proxy_metrics(c)})
            if (i+1)%25==0: print(f'{name}: {i+1}/{len(samples)}', flush=True)
        total = sum_counts([r['counts'] for r in rows])
        per_group = {}
        for group in sorted({r['group'] for r in rows}):
            members = [r for r in rows if r['group']==group]
            c = sum_counts([r['counts'] for r in members])
            per_group[group] = {'n':len(members),'counts':c,'metrics':proxy_metrics(c)}
        report['models'][name] = {'counts':total, 'metrics':proxy_metrics(total), 'per_group':per_group,
             'parameters':sum(p.numel() for p in model.model.parameters()),
             'parameter_dtype':str(next(model.model.parameters()).dtype),
             'samples':rows, 'timings':timings[name], 'input_nchw':timings[name][0]['input_nchw'],
             'latency':{stage:summarize([r[stage] for r in timings[name]]) for stage in ['preprocess_ms','forward_ms','postprocess_ms','pipeline_ms']}}
    assert all(len(m['samples'])==len(samples) for m in report['models'].values())
    report['complete'] = True
    (args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    # Failure artifacts are generated AFTER timing and metrics, into ignored output.
    failures = set()
    for m in report['models'].values():
        for key,reverse in [('grass_false_positive_rate',True),('grass_false_negative_rate',True),('grass_iou',False)]:
            failures.update(r['id'] for r in sorted(m['samples'],key=lambda r:r['metrics'][key] if r['metrics'][key] is not None else (-1 if reverse else 2),reverse=reverse)[:3])
    for s in samples:
        if s['id'] not in failures: continue
        with Image.open(args.root/s['image']) as image: rgb=image.convert('RGB')
        truth=read_binary(args.root/s['mask'],SIZE)
        panels=[rgb.resize(SIZE), Image.fromarray(np.where(truth==255,127,truth*255).astype(np.uint8))]
        for name,model in models.items():
            pred,_=model.predict(rgb)
            err=np.zeros((SIZE[1],SIZE[0],3),np.uint8)
            err[(pred==1)&(truth==0)]=(255,40,40)
            err[(pred==0)&(truth==1)]=(40,80,255)
            err[truth==255]=(127,127,127)
            panels.extend([Image.fromarray(pred*255),Image.fromarray(err)])
        canvas=Image.new('RGB',(SIZE[0]*3,SIZE[1]*2+40),'white')
        for i,p in enumerate(panels): canvas.paste(p.convert('RGB'),((i%3)*SIZE[0],40+(i//3)*SIZE[1]))
        ImageDraw.Draw(canvas).text((8,8),s['id']+' | RGB, GT, LR pred / LR error, B0 pred, B0 error; red FP blue FN gray ignore',fill='black')
        canvas.save(args.output/(s['id']+'-failure.jpg'))
    print(json.dumps({n:{'metrics':m['metrics'],'latency':m['latency']} for n,m in report['models'].items()},indent=2),flush=True)


if __name__ == '__main__':
    main()

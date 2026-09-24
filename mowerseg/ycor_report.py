"""Read the frozen, complete YCOR comparison without requiring raw images."""
import hashlib
import json
from pathlib import Path

import numpy as np

from mowerseg.ycor import proxy_metrics, sum_counts

ROOT = Path(__file__).resolve().parents[1]


def summary(directory=ROOT / 'evaluation/ycor'):
    directory = Path(directory)
    report = json.loads((directory / 'results.json').read_text())
    manifest_bytes = (directory / 'split.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = {s['id']: s for s in manifest['samples'] if s['split'] == 'test'}
    if (not report.get('complete') or not expected
            or report['sample_count'] != len(expected)
            or report['frozen']['manifest_sha256'] != hashlib.sha256(manifest_bytes).hexdigest()
            or set(report['models']) != {'lraspp', 'b0'}):
        raise ValueError('Incomplete or mismatched frozen YCOR report')
    for model in report['models'].values():
        rows = model['samples']
        if len(rows) != len(expected) or {s['id'] for s in rows} != set(expected):
            raise ValueError('Incomplete sample coverage')
        for row in rows:
            original = expected[row['id']]
            if (any(row[k] != original[k] for k in ('image_sha256', 'mask_sha256'))
                    or row['split_group'] != original['group']
                    or row['group'] != original['report_group']):
                raise ValueError('Sample provenance mismatch')
            if row['metrics'] != proxy_metrics(row['counts']):
                raise ValueError('Sample metrics mismatch')
        if model['counts'] != sum_counts([s['counts'] for s in rows]):
            raise ValueError('Aggregate count mismatch')
        if model['metrics'] != proxy_metrics(model['counts']):
            raise ValueError('Aggregate metrics mismatch')
        timing = model['timings']
        expected_n = report['timing_protocol']['images'] * report['timing_protocol']['repeats_per_image']
        if len(timing) != expected_n:
            raise ValueError('Timing coverage mismatch')
        for key, latency in model['latency'].items():
            values = [t[key] for t in timing]
            if (latency['n'] != len(timing) or not np.isfinite(values).all()
                    or min(values) < 0 or latency['p50_ms'] != float(np.percentile(values, 50))
                    or latency['p95_ms'] != float(np.percentile(values, 95))):
                raise ValueError('Timing quantile mismatch')
        for t in timing:
            if abs(t['pipeline_ms'] - sum(t[k] for k in ('preprocess_ms', 'forward_ms', 'postprocess_ms'))) > 1e-6:
                raise ValueError('Pipeline timing boundary mismatch')
        groups = {s['group'] for s in rows}
        if set(model['per_group']) != groups:
            raise ValueError('Group coverage mismatch')
        for group, values in model['per_group'].items():
            members = [s for s in rows if s['group'] == group]
            if (values['n'] != len(members)
                    or values['counts'] != sum_counts([s['counts'] for s in members])
                    or values['metrics'] != proxy_metrics(values['counts'])):
                raise ValueError('Group metrics mismatch')
    return {
        'available': True, 'sample_count': report['sample_count'],
        'scope': report['scope'], 'frozen': report['frozen'],
        'hardware': report['hardware'], 'software': report['software'],
        'timing_protocol': report['timing_protocol'], 'boundary': report['boundary'],
        'models': [{'id': name, 'name': 'YCOR LR-ASPP' if name == 'lraspp' else 'SegFormer-B0 (ADE20K)',
                    **{k: m[k] for k in ('metrics', 'counts', 'per_group', 'latency', 'input_nchw')}}
                   for name, m in report['models'].items()],
    }


def demo_samples():
    record = json.loads((ROOT / 'evaluation/ycor/demo-samples.json').read_text())
    return [s for s in record['samples'] if (ROOT / 'data/ycor-demo' / s['file']).is_file()]


def demo_image(sample_id):
    from PIL import Image
    sample = next((s for s in demo_samples() if s['id'] == sample_id), None)
    if sample is None:
        raise FileNotFoundError('YCOR 本机样例缺失')
    path = ROOT / 'data/ycor-demo' / sample['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != sample['image_sha256']:
        raise ValueError('YCOR 样例摘要不匹配')
    with Image.open(path) as image:
        return image.convert('RGB')

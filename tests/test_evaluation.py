from pathlib import Path
import csv
import numpy as np
from PIL import Image
import pytest
import torch

from mowerseg.core.frame import Frame
from mowerseg.evaluate import counts, metrics, boundary, dilate
from mowerseg.models.registry import load_model_config
from mowerseg.tasks.segmentation.preprocess import SegmentationPreprocessor
from mowerseg.tasks.segmentation.postprocess import logits_to_label_mask
from mowerseg.taxonomy import load_taxonomy
from tools.prepare_grassseghb import select_samples

ROOT = Path(__file__).resolve().parents[1]


def test_metrics_exact_counts_and_undefined():
    c = counts(np.array([[1, 1], [0, 0]]), np.array([[1, 0], [1, 0]]), 0)
    assert [c[k] for k in ('tp', 'fp', 'tn', 'fn')] == [1, 1, 1, 1]
    m = metrics(c)
    assert m['mowable_iou'] == 1/3
    assert m['nonmowable_false_positive_rate'] == .5
    assert m['predicted_mowable_error_fraction'] == .5
    empty = metrics(counts(np.zeros((4, 4)), np.zeros((4, 4))))
    assert empty['mowable_iou'] is None
    assert empty['nonmowable_false_positive_rate'] == 0
    assert empty['boundary_f1'] is None
    with pytest.raises(ValueError, match='0/1'):
        counts(np.zeros((1, 1)), np.array([[255]]))
    with pytest.raises(ValueError, match='geometry'):
        counts(np.zeros((1, 2)), np.zeros((2, 1)))


def test_boundary_shift_and_perimeter():
    gt = np.zeros((20, 30), dtype=bool); gt[:, 10:] = True
    pred = np.zeros_like(gt); pred[:, 12:] = True
    assert metrics(counts(pred, gt, 2))['boundary_f1'] == 1
    assert metrics(counts(pred, gt, 0))['boundary_f1'] == 0
    assert boundary(gt).sum() == 40
    assert not boundary(np.ones((3, 3))).any()
    single = np.zeros((9, 9), dtype=bool); single[4, 4] = True
    assert dilate(single, 3).sum() == 49


def test_ade20k_offset_ignore_and_product_mapping():
    with (ROOT/'mowerseg/models/ade20k_classes.csv').open() as f:
        classes = list(csv.DictReader(f))
    assert len(classes) == 150
    assert classes[9]['Name'] == 'grass' and int(classes[9]['Idx']) == 10
    # Annotation 0 = unlabeled, 1..150 = classes; outputs are 0..149.
    raw_labels = np.array([0, 1, 10, 150], dtype=np.int16)
    assert (raw_labels - 1).tolist() == [-1, 0, 9, 149]
    tax = load_taxonomy(ROOT/'configs/mower_seg.yaml')
    assert np.flatnonzero(tax.source_lookup == 1).tolist() == [9]
    assert tax.remap(np.array([0, 9, 4, 13, 255], dtype=np.uint8)).tolist() == [7, 1, 2, 3, 7]


@pytest.mark.parametrize('model', ['segformer_b0_ade20k', 'mit_resnet18_ade20k'])
def test_non_square_edges_and_normalization(model):
    array = np.full((1068, 1600, 3), 255, dtype=np.uint8)
    array[:, :160] = (255, 0, 0); array[:, -160:] = (0, 0, 255)
    pre = SegmentationPreprocessor(load_model_config(model))
    pixels = pre(Frame.from_image(Image.fromarray(array)))['pixel_values']
    expected = (512, 512) if model.startswith('segformer') else (600, 904)
    assert pixels.shape == (1, 3, *expected)
    np.testing.assert_allclose(pixels[0, :, 100, 0], (np.array([1, 0, 0])-[.485, .456, .406])/[.229, .224, .225], atol=1e-5)
    assert pixels[0, 2, 100, -1] > 2
    # Restore synthetic predictions derived from actual preprocessed colour bands.
    logits = pixels[:, [0, 1, 2]]
    restored = logits_to_label_mask(logits, array.shape[:2])
    assert np.all(restored[:, :100] == 0)
    assert np.all(restored[:, -100:] == 2)


def test_chunked_restore_matches_full_resize():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(1, 21, 7, 13)).astype(np.float32)
    expected = torch.nn.functional.interpolate(torch.from_numpy(logits), (31, 79), mode='bilinear', align_corners=False).argmax(1)[0].numpy()
    np.testing.assert_array_equal(logits_to_label_mask(logits, (31, 79)), expected)
    assert not logits_to_label_mask(np.zeros((1, 17, 2, 3), dtype=np.float32), (8, 12)).any()


def test_sampling_is_seeded_grouped_and_unique():
    from zipfile import ZipInfo
    infos = [ZipInfo(f'images/{g}G{n:04d}.jpg') for g in range(3) for n in range(100)]
    a = select_samples(infos, 42, 16)
    assert a == select_samples(reversed(infos), 42, 16)
    assert a != select_samples(infos, 43, 16)
    assert len(a) == len({s['image'] for s in a}) == 48
    assert all(sum(s['garden'] == str(g) for s in a) == 16 for g in range(3))


def test_mit_refuses_missing_or_incomplete_checkpoints(tmp_path):
    from mowerseg.models.mit_resnet import load_mit_resnet18
    with pytest.raises(FileNotFoundError, match='download_mit_weights'):
        load_mit_resnet18(tmp_path)
    (tmp_path/'encoder_epoch_20.pth').write_bytes(b'not a segmentation checkpoint')
    with pytest.raises(ValueError, match='SHA-256'):
        load_mit_resnet18(tmp_path)

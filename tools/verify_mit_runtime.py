"""Compare the vendored inference subset against a separately cloned official source."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from mowerseg.core.frame import Frame
from mowerseg.models.mit_resnet import load_mit_resnet18
from mowerseg.models.registry import load_model_config
from mowerseg.tasks.segmentation.preprocess import SegmentationPreprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path(".cache/mit-resnet18"))
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if revision != "8f27c9b97d2ca7c6e05333d5766d144bf7d8c31b":
        raise ValueError("Use the documented official source revision")
    sys.path.insert(0, str(args.source.resolve()))
    from mit_semseg.models import ModelBuilder
    from mit_semseg.dataset import TestDataset

    torch.set_num_threads(4)
    torch.manual_seed(20260921)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ours = load_mit_resnet18(args.weights).eval().to(device)
    encoder = ModelBuilder.build_encoder("resnet18dilated", weights=str(args.weights/"encoder_epoch_20.pth")).eval().to(device)
    decoder = ModelBuilder.build_decoder("ppm_deepsup", fc_dim=512, num_class=150,
                                        weights=str(args.weights/"decoder_epoch_20.pth"), use_softmax=True).eval().to(device)
    # Compare same input tensor, including the original synchronized BN eval path.
    results = []
    for image in ["assets/samples/lawn_path.jpg", "assets/samples/garden_walk.jpg"]:
        dataset = TestDataset([{"fpath_img": image}], SimpleNamespace(imgSizes=[600], imgMaxSize=1000, padding_constant=8))
        reference = dataset[0]["img_data"][0]
        actual = SegmentationPreprocessor(load_model_config("mit_resnet18_ade20k"))(Frame.from_image(image))["pixel_values"]
        np.testing.assert_array_equal(actual, reference.numpy())
        values = reference.to(device)
        target = (137, 211)
        with torch.inference_mode():
            expected = decoder(encoder(values, return_feature_maps=True), segSize=target)
            probability = torch.nn.functional.interpolate(ours(values), target, mode="bilinear", align_corners=False).softmax(1)
        torch.testing.assert_close(probability, expected, atol=1e-6, rtol=1e-5)
        assert torch.equal(probability.argmax(1), expected.argmax(1))
        results.append({"image": image, "input_shape": list(values.shape),
                        "max_probability_difference": float((probability-expected).abs().max()), "argmax_equal": True})
    print(json.dumps({"official_revision": revision, "device": device, "comparisons": results}, indent=2))


if __name__ == "__main__":
    main()

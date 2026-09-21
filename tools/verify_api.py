"""Exercise the real running production frontend proxy and segmentation models."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import urllib.request

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default="http://127.0.0.1:43129")
    parser.add_argument("--output", type=Path, default=Path("outputs/api-validation.json"))
    args = parser.parse_args()
    results = []
    with urllib.request.urlopen(args.origin + "/api/models", timeout=120) as response:
        models = json.load(response)["models"]
    for model in models:
        for sample in ("lawn_path", "garden_walk", "backyard_house"):
            req = urllib.request.Request(f"{args.origin}/api/infer-sample/{sample}?model={model['id']}", data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=180) as response:
                payload = json.load(response)
            assert payload["model"]["id"] == model["id"]
            sizes = [Image.open(BytesIO(base64.b64decode(payload[k].split(",", 1)[1]))).size for k in ("input", "mask", "overlay")]
            assert len(set(sizes)) == 1
            results.append({"model": model["id"], "sample": sample, "preview_size": sizes[0], "input_shape": payload["stats"]["input_shape"], "ok": True})
        boundary = "mower-validation-boundary"
        data = Path("assets/samples/garden_walk.jpg").read_bytes()
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"upload.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode()
                + data + f"\r\n--{boundary}--\r\n".encode())
        req = urllib.request.Request(f"{args.origin}/api/infer?model={model['id']}", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=180) as response:
            payload = json.load(response)
        assert payload["model"]["id"] == model["id"]
        sizes = [Image.open(BytesIO(base64.b64decode(payload[k].split(",", 1)[1]))).size for k in ("input", "mask", "overlay")]
        assert len(set(sizes)) == 1
        results.append({"model": model["id"], "upload": "garden_walk.jpg", "preview_size": sizes[0], "input_shape": payload["stats"]["input_shape"], "ok": True})
        print(f"Verified {model['id']}: 3 samples + multipart upload", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()

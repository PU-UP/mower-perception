"""Download the complete official ADE20K ResNet18+PPM weights, with pinned hashes."""
from pathlib import Path
import argparse
import hashlib
import urllib.request

BASE = "http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet18dilated-ppm_deepsup"
from mowerseg.models.mit_resnet import WEIGHTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(".cache/mit-resnet18"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in WEIGHTS.items():
        dest = args.directory / name
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == digest:
            print(f"Verified {name}")
            continue
        # Never expose a partial checkpoint to the runtime; reruns resume the .part.
        part = dest.with_suffix(".pth.part")
        start = part.stat().st_size if part.exists() else 0
        if start > size:
            raise ValueError(f"Oversized partial file: {part}")
        if start < size:
            request = urllib.request.Request(f"{BASE}/{name}", headers={"Range": f"bytes={start}-"})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise ValueError("Expected resumable HTTP range response")
                with part.open("ab") as output:
                    while block := response.read(1024*1024):
                        output.write(block)
        if part.stat().st_size != size or hashlib.sha256(part.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Weight checksum failed: {part}; inspect/remove this partial file before retrying")
        part.replace(dest)
        print(f"Downloaded and verified {name}")


if __name__ == "__main__":
    main()

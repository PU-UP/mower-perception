"""Fetch a deterministic garden-stratified GrassSegHB subset, not the 27 GiB ZIP."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import re
import shutil
import struct
import time
import urllib.request
import zipfile
import zlib

URL = "https://www.informatik.uni-bremen.de/agebv2/pub/grassseghb/grassseghb.zip"


def byte_range(start, length):
    for attempt in range(3):
        try:
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start+length-1}"})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise ValueError("Server did not honor byte range; refusing full archive")
                data = response.read(length)
            if len(data) != length:
                raise ValueError("Truncated download")
            return data
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


class RemoteZip(io.RawIOBase):
    def __init__(self):
        with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=30) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
        self.pos = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = byte_range(self.pos, n)
        self.pos += len(data)
        return data


def select_samples(infos, seed, per_garden):
    """One random frame in each chronological bin per garden; no split mixing."""
    groups = {}
    for info in infos:
        match = re.fullmatch(r"images/(\d+)G(\d+)\.jpg", info.filename)
        if match:
            groups.setdefault(match[1], []).append((int(match[2]), info.filename))
    rng = random.Random(seed)
    selected = []
    for garden in sorted(groups, key=int):
        frames = sorted(groups[garden])
        if len(frames) < per_garden:
            raise ValueError(f"Garden {garden} has too few frames")
        for i in range(per_garden):
            section = frames[i*len(frames)//per_garden:(i+1)*len(frames)//per_garden]
            _, image = rng.choice(section)
            selected.append({"garden": garden, "image": image,
                             "mask": f"annotations/{Path(image).stem}_mask.png"})
    return selected


def fetch_member(info, root):
    # Only known image/mask names can be written; never extract arbitrary ZIP paths.
    if not re.fullmatch(r"(?:images/\d+G\d+\.jpg|annotations/\d+G\d+_mask\.png)", info.filename):
        raise ValueError(f"Unexpected member {info.filename}")
    dest = root / info.filename
    if dest.exists():
        data = dest.read_bytes()
        if len(data) == info.file_size and zlib.crc32(data) == info.CRC:
            return hashlib.sha256(data).hexdigest()
    header = byte_range(info.header_offset, 30)
    if header[:4] != b"PK\x03\x04":
        raise ValueError("Invalid ZIP local header")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    raw = byte_range(info.header_offset + 30 + name_len + extra_len, info.compress_size)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        data = raw
    else:
        raise ValueError("Unsupported ZIP compression")
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ValueError(f"CRC/size mismatch: {info.filename}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(dest)
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/grassseghb"))
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/grassseghb-256.json"))
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--per-garden", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--download", action="store_true", help="Without this flag, inspect sizes and save planned manifest only")
    args = parser.parse_args()
    if args.per_garden <= 0:
        parser.error("--per-garden must be positive")
    remote = RemoteZip()
    with zipfile.ZipFile(remote) as archive:
        infos = {i.filename: i for i in archive.infolist()}
    samples = select_samples(infos.values(), args.seed, args.per_garden)
    members = [infos[s[k]] for s in samples for k in ("image", "mask")]
    size = sum(i.file_size for i in members)
    args.root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.root).free
    manifest = {"source": URL, "archive_bytes": remote.size, "etag": remote.etag,
                "seed": args.seed, "per_garden": args.per_garden,
                "sampling": "sorted numeric frame bins; seeded random frame per bin per garden",
                "split": "exploratory_only; reserve whole gardens before any training",
                "license": "No explicit license on landing page or archive; commercial rights unconfirmed",
                "download_complete": False, "samples": samples}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest.exists():
        existing = json.loads(args.manifest.read_text())
        identity = lambda m: (m["source"], m["archive_bytes"], m["etag"], m["seed"],
                              [(s["garden"], s["image"], s["mask"]) for s in m["samples"]])
        if identity(existing) != identity(manifest):
            raise ValueError("Existing manifest identifies a different subset/archive; choose a new --manifest path")
        # Inspecting or an interrupted rerun must not destroy frozen hashes.
    else:
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(samples)} pairs; {size/2**20:.1f} MiB selected; {free/2**30:.1f} GiB free", flush=True)
    if not args.download:
        return
    if free < size * 2:
        raise ValueError("Insufficient space for data and temporary downloads")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, digest in enumerate(pool.map(lambda i: fetch_member(i, args.root), members)):
            samples[index//2][("image_sha256", "mask_sha256")[index % 2]] = digest
            if index % 32 == 31:
                print(f"Downloaded {index+1}/{len(members)} members", flush=True)
    manifest["download_complete"] = True
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

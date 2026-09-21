"""Validate the frozen subset's labels/geometry/hashes and build a visual audit sheet."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mowerseg.evaluate import load_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/grassseghb"))
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/grassseghb-256.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/grass-audit"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not manifest.get("download_complete"):
        raise ValueError("Dataset download is not complete")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in manifest["samples"]:
        image, mask = load_pair(args.root, s)
        thumbnail = image.resize((160, 120))
        luminance = np.asarray(thumbnail.convert("L"))
        rows.append({"image": s["image"], "garden": s["garden"], "size": list(image.size),
                     "labels": np.unique(mask).tolist(), "mowable_fraction": float(mask.mean()),
                     "mean_luminance_0_255": float(luminance.mean()),
                     "luminance_p10": float(np.percentile(luminance, 10)),
                     "luminance_p90": float(np.percentile(luminance, 90))})
    # Light/dark representatives per group selected by image luminance, not model quality.
    selected = []
    for garden in sorted({r["garden"] for r in rows}, key=int):
        ordered = sorted([r for r in rows if r["garden"] == garden], key=lambda r:r["mean_luminance_0_255"])
        selected.extend([ordered[0], ordered[-1]])
    for start in range(0, len(selected), 8):
        sheet = Image.new("RGB", (1280, 4*270), "white")
        for i, row in enumerate(selected[start:start+8]):
            sample = next(s for s in manifest["samples"] if s["image"] == row["image"])
            image, mask = load_pair(args.root, sample)
            x, y = (i%2)*640, (i//2)*270
            sheet.paste(image.resize((320, 240)), (x, y+30))
            sheet.paste(Image.fromarray(mask*255).resize((320, 240), Image.Resampling.NEAREST), (x+320, y+30))
            ImageDraw.Draw(sheet).text((x+4,y+6), f"{Path(row['image']).stem} L={row['mean_luminance_0_255']:.1f} | RGB / GT", fill="black")
        sheet.save(args.output/f"audit-{start//8}.jpg")
    summary = {"n": len(rows), "gardens": len({r['garden'] for r in rows}),
               "sizes": sorted({tuple(r['size']) for r in rows}),
               "labels": sorted({v for r in rows for v in r['labels']}),
               "luminance_range": [min(r['mean_luminance_0_255'] for r in rows), max(r['mean_luminance_0_255'] for r in rows)],
               "mowable_fraction_range": [min(r['mowable_fraction'] for r in rows), max(r['mowable_fraction'] for r in rows)],
               "note": "Luminance is a coverage proxy, not a verified illumination annotation; contact sheets need human review",
               "samples": rows}
    (args.output/'audit.json').write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k != 'samples'}, indent=2))


if __name__ == "__main__":
    main()

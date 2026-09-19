from __future__ import annotations

import argparse
import json
from pathlib import Path

from mowerseg.infer import InferenceEngine


def default_config() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "mower_seg.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="割草机可通行语义分割推理")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("-c", "--config", default=str(default_config()), help="类别与模型配置")
    parser.add_argument("-o", "--output", default="outputs", help="结果输出目录")
    parser.add_argument("--stem", default=None, help="输出文件名前缀")
    args = parser.parse_args()

    image_path = Path(args.image)
    engine = InferenceEngine(args.config)
    result = engine.predict(image_path)
    stem = args.stem or image_path.stem
    paths = engine.save(result, args.output, stem)
    print(json.dumps({"stats": result.stats, "files": paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

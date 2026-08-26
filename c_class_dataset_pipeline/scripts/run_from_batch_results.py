#!/usr/bin/env python
"""生产用入口（离线版）：复用 vlm-bbox-labeling/batch_run.py 已经落盘的 raw/*.json，
不用重新调用一遍 VLM 服务。

适合大批量场景：先用 batch_run.py 把全部图片跑完、人工按 verify/ 抽查过一轮，
确认没问题后再用这个脚本把 raw/*.json 转成 C 类数据集，比每次都重新调服务更稳妥
（不会因为服务抖动导致大批量任务从头重跑）。

用法：
  python vlm-bbox-labeling/batch_run.py --input ./images --output ./results
  python scripts/run_from_batch_results.py --raw-dir ./results/raw --images-dir ./images \
      --output output/c_class_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.bbox_service_adapter import detection_result_to_annotation_payload  # noqa: E402
from pipeline.build_dataset import build_dataset, print_report  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def iter_payloads_from_raw(raw_dir: Path, images_dir: Path):
    for raw_path in sorted(raw_dir.glob("*.json")):
        result = json.loads(raw_path.read_text(encoding="utf-8"))
        image_name = result.get("image", {}).get("name") or f"{raw_path.stem}.jpg"
        image_path = find_image(images_dir, Path(image_name).stem) or (images_dir / image_name)
        payload = detection_result_to_annotation_payload(result, image_path, sample_id=raw_path.stem)
        if payload is not None:
            yield payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True, help="batch_run.py 输出的 results/raw 目录")
    ap.add_argument("--images-dir", type=Path, required=True, help="原始图片目录")
    ap.add_argument("--output", type=Path, default=Path("output/c_class_dataset.jsonl"))
    args = ap.parse_args()

    if not args.raw_dir.exists():
        raise SystemExit(f"找不到 raw 目录：{args.raw_dir}")

    stats = build_dataset(iter_payloads_from_raw(args.raw_dir, args.images_dir), args.output)
    print_report(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

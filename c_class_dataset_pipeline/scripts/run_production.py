#!/usr/bin/env python
"""生产用入口：对接真正的 vlm-bbox-labeling 服务，产出满足 C 类指标的数据集。

前提：
  1. vlm-bbox-labeling 已按其 README 起好服务（docker compose up -d --build），
     `classes.yaml` 换成你们自己的 347 类业务类别表（不是 COCO128 那份）。
  2. `curl http://<base-url>/health` 返回 classes_loaded == 347。

用法：
  python scripts/run_production.py --images-dir /path/to/images \
      --base-url http://localhost:8000 --output output/c_class_dataset.jsonl

规模不够 10 万条时，脚本会在报告里给出还需要多少张源图片；
也可以直接跑 scripts/estimate_scale.py 提前算好要收集/标注多少张图。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.bbox_service_adapter import iter_annotation_payloads_via_service  # noqa: E402
from pipeline.build_dataset import build_dataset, print_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", type=Path, required=True, help="待标注图片目录")
    ap.add_argument("--base-url", default="http://localhost:8000", help="vlm-bbox-labeling 服务地址")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--output", type=Path, default=Path("output/c_class_dataset.jsonl"))
    args = ap.parse_args()

    if not args.images_dir.exists():
        raise SystemExit(f"找不到图片目录：{args.images_dir}")

    payloads = iter_annotation_payloads_via_service(
        args.images_dir, base_url=args.base_url, timeout=args.timeout
    )
    stats = build_dataset(payloads, args.output)
    print_report(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

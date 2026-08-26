#!/usr/bin/env python
"""估算要凑够 C 类指标要求的 ≥10 万条样本，大概需要多少张源图片。

两种用法：

1. 已经跑过一批（比如 coco128 联调），从产出的 JSONL 直接算实际的"每图样本数"，
   再推算目标规模需要的图片数：

       python scripts/estimate_scale.py --from-jsonl examples/coco128_c_class_sample.jsonl \
           --distinct-images 128

2. 还没有任何产出，纯按经验参数估算（每图平均目标框数、开启的任务类型数）：

       python scripts/estimate_scale.py --avg-boxes-per-image 4 --avg-labels-per-image 2

任务类型对样本数的贡献（对应 build_dataset.py 里默认打开的 4 种类型）：
    detect_all        : 每图固定 1 条
    detect_label       : 每图约等于"图内出现的类别数"条
    ground_single      : 每图约等于"图内目标框数"条
    vlm_grounding_qa   : 每图 qa_per_image_min ~ qa_per_image_max 条（默认 3~5，取中间值）
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TARGET_SAMPLE_COUNT = 100_000


def estimate_samples_per_image(avg_boxes: float, avg_labels: float, qa_mid: float = 4.0) -> float:
    return 1.0 + avg_labels + avg_boxes + qa_mid  # detect_all + detect_label + ground_single + vlm_grounding_qa


def from_jsonl(path: Path, distinct_images: int) -> float:
    count = sum(1 for _ in path.open("r", encoding="utf-8"))
    return count / distinct_images if distinct_images else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=TARGET_SAMPLE_COUNT)
    ap.add_argument("--from-jsonl", type=Path, help="已产出的 C 类 JSONL 文件路径")
    ap.add_argument("--distinct-images", type=int, help="上面 JSONL 里对应的源图片数量（配合 --from-jsonl）")
    ap.add_argument("--avg-boxes-per-image", type=float, default=4.0, help="纯估算模式：平均每图目标框数")
    ap.add_argument("--avg-labels-per-image", type=float, default=2.0, help="纯估算模式：平均每图出现的类别数")
    ap.add_argument("--qa-mid", type=float, default=4.0, help="纯估算模式：vlm_grounding_qa 每图取的中间条数")
    args = ap.parse_args()

    if args.from_jsonl:
        if not args.distinct_images:
            raise SystemExit("使用 --from-jsonl 时必须同时给出 --distinct-images")
        ratio = from_jsonl(args.from_jsonl, args.distinct_images)
        source = f"实测（{args.from_jsonl}，{args.distinct_images} 张图）"
    else:
        ratio = estimate_samples_per_image(args.avg_boxes_per_image, args.avg_labels_per_image, args.qa_mid)
        source = "经验估算"

    images_needed = math.ceil(args.target / ratio) if ratio > 0 else None

    print(f"每图平均样本数   : {round(ratio, 2)}  （来源：{source}）")
    print(f"目标样本数        : {args.target}")
    print(f"需要源图片数（约） : {images_needed}")
    print()
    print("说明：这里的“图片”指已完成人工复核、可进训练集的有效标注图片。")
    print("COCO128 这类开源数据只用于管道联调，不计入这个数量——它是日常类别，")
    print("跟专业领域数据差别很大（见 vlm-bbox-labeling/LOCAL_TESTING.md）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

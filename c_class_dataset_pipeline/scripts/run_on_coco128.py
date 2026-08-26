#!/usr/bin/env python
"""管道联调：用 COCO128 把"标注 -> SFT 语料 -> C 类样本"整条链路跑通。

COCO128 是 vlm-bbox-labeling/LOCAL_TESTING.md 里提到的开源数据集
（https://www.kaggle.com/datasets/ultralytics/coco128 ，128 张图、80 类，
自带 YOLO 格式标准答案），已经放在 vlm-bbox-labeling/coco128/ 下，
配套的 80 类类别表就是 vlm-bbox-labeling/classes.yaml。

用它跑通的意义和边界，vlm-bbox-labeling 的文档已经说得很清楚：
    "COCO 是日常类别，跟专业领域数据差别很大，只能用来验证管道通不通、
     看模型的基础定位能力，不能代表专业领域的表现"

所以这一步不调用真正的 VLM 服务（那需要能连到内网的 Qwen3.6 部署），
而是直接读 COCO128 自带的标准答案（labels/*.txt），验证：

    YOLO 标注 -> 坐标换算 -> 4 种任务类型的 SFT 样本 -> 描述语句 -> C 类样本结构

这条链路本身是否正确、字段是否齐全。生产环境把 yolo_gt_adapter 换成
bbox_service_adapter（真调 VLM 服务）+ 你们自己的 347 类 classes.yaml 即可。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_dataset import build_dataset, print_report  # noqa: E402
from pipeline.deps import BBOX_LABELING_DIR, load_class_table  # noqa: E402
from pipeline.yolo_gt_adapter import iter_annotation_payloads  # noqa: E402

DEFAULT_LABELS_DIR = BBOX_LABELING_DIR / "coco128" / "labels" / "train2017"
DEFAULT_IMAGES_DIR = BBOX_LABELING_DIR / "coco128" / "images" / "train2017"
DEFAULT_CLASSES_YAML = BBOX_LABELING_DIR / "classes.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    ap.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    ap.add_argument("--classes-yaml", type=Path, default=DEFAULT_CLASSES_YAML)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "examples" / "coco128_c_class_sample.jsonl")
    args = ap.parse_args()

    if not args.labels_dir.exists():
        raise SystemExit(f"找不到 COCO128 标注目录：{args.labels_dir}")

    table = load_class_table(str(args.classes_yaml))
    payloads = iter_annotation_payloads(args.labels_dir, args.images_dir, table)

    stats = build_dataset(payloads, args.output)
    print_report(stats)
    print(f"\n样例已写入：{args.output}（可用 head -3 看几条真实样本）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

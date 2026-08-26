#!/usr/bin/env python
"""统计标注数据的分布，用于确定质量过滤阈值和推算规模需求。

规划文档里那三个过滤阈值（每图框数上限、框短边像素、框面积占比）
就是用这个脚本在 COCO128 上跑出来的分位数定的。

拿到正式的业务标注数据后，用同样的脚本重跑一遍，按真实分布重新定阈值 ——
COCO 是日常密集场景，专业领域数据的目标密度和尺寸分布通常很不一样，
直接套用 COCO 的阈值会误伤。

用法：
    python scripts/analyze_dataset.py                      # 默认跑 COCO128
    python scripts/analyze_dataset.py --labels-dir ... --images-dir ... --classes-yaml ...
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.deps import BBOX_LABELING_DIR, load_class_table, read_image_size  # noqa: E402
from pipeline.yolo_gt_adapter import find_image_for_label, parse_yolo_txt  # noqa: E402

# 每图框数「上限」已废弃为质量规则，只保留一个极宽松的完整性检查：
# 超过这个数几乎必然是标注文件损坏，而不是真的有这么多目标。
#
# 为什么废弃：航拍/密集场景天生每图几十上百个目标（VisDrone 每图均 70 个、
# 最多 317 个），但「框多」本身不是质量问题 —— 一张有 70 辆车的停车场航拍图，
# 每辆车都标得很准。实测在 VisDrone 上启用「≤20 框/图」会跳过 480/548 张图
# （88%），样本量从 1059 条掉到 134 条，而困难目标占比两者都是 10%。
# 密集问题应该交给 difficulty.py 里的 density/referring 维度【逐目标】判断，
# 比整张图一刀切精细得多。
SANITY_MAX_BOXES = 1000
MIN_SHORT_SIDE_PX = 16
MIN_AREA_RATIO = 0.001
# 框占整图面积超过这个比例时剔除。实测业务数据里存在占满全图的框
# （如某标注文件里 99.7% 面积的框），这类框做视觉定位样本毫无意义 ——
# 「请定位图中的 X」答案是整张图，模型学不到定位，反而被教着输出满图框。
MAX_AREA_RATIO = 0.5
SAMPLES_PER_IMAGE_CAP = 8
TARGET_SAMPLES = 100_000

BOXES_PER_IMAGE_CANDIDATES = (10, 15, 20, 25, 30)
SHORT_SIDE_CANDIDATES = (8, 12, 16, 24)
AREA_RATIO_CANDIDATES = (0.0005, 0.001, 0.002, 0.005)
AREA_MAX_CANDIDATES = (0.3, 0.5, 0.7, 0.9)


def pct(values, p):
    """第 p 百分位。values 需已排序且非空。"""
    return statistics.quantiles(values, n=100)[p - 1]


def collect(labels_dir: Path, images_dir: Path, table):
    """扫描目录，返回 [(图片名, 宽, 高, [框, ...]), ...]。框带 cx/cy/w/h 归一化坐标。"""
    out = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = find_image_for_label(label_path, images_dir)
        if image_path is None:
            continue
        rows = parse_yolo_txt(label_path, table)
        if not rows:
            continue
        w, h = read_image_size(image_path)
        out.append((image_path.name, w, h, rows))
    return out


def short_side_px(row, img_w: int, img_h: int) -> float:
    return min(row["w"] * img_w, row["h"] * img_h)


def area_ratio(row) -> float:
    return row["w"] * row["h"]


def report(images) -> None:
    box_counts = sorted(len(rows) for _, _, _, rows in images)
    total_boxes = sum(box_counts)
    n_images = len(images)

    all_short_sides = sorted(
        short_side_px(r, w, h) for _, w, h, rows in images for r in rows
    )
    all_areas = sorted(area_ratio(r) for _, _, _, rows in images for r in rows)

    print("=" * 68)
    print(f"图片数 {n_images}    标注框数 {total_boxes}    每图平均 {total_boxes / n_images:.2f} 个框")
    print("=" * 68)

    print("\n【每图框数分布】")
    print(f"  中位数 {statistics.median(box_counts):.0f}   "
          f"p75 {pct(box_counts, 75):.0f}   p90 {pct(box_counts, 90):.0f}   "
          f"p95 {pct(box_counts, 95):.0f}   最大 {max(box_counts)}")
    print("\n  （仅供参考，不作为过滤规则 —— 密集度由 difficulty.py 逐目标判断）")
    for cap in BOXES_PER_IMAGE_CANDIDATES:
        kept = [c for c in box_counts if c <= cap]
        print(f"    若限制 ≤{cap:>3} 框/图  ->  仅保留 {len(kept):>4}/{n_images} 图 "
              f"({len(kept) / n_images * 100:>3.0f}%)、{sum(kept):>5}/{total_boxes} 框 "
              f"({sum(kept) / total_boxes * 100:>3.0f}%)")

    print("\n【框尺寸分布】")
    print(f"  短边像素   p5 {pct(all_short_sides, 5):.1f}   p10 {pct(all_short_sides, 10):.1f}   "
          f"p25 {pct(all_short_sides, 25):.1f}   中位数 {statistics.median(all_short_sides):.1f}")
    print(f"  面积占比   p5 {pct(all_areas, 5) * 100:.3f}%   p10 {pct(all_areas, 10) * 100:.3f}%   "
          f"p25 {pct(all_areas, 25) * 100:.3f}%   中位数 {statistics.median(all_areas) * 100:.3f}%")

    print("\n  规则二 · 剔除过小的目标：")
    for thr in SHORT_SIDE_CANDIDATES:
        drop = sum(1 for v in all_short_sides if v < thr)
        print(f"    短边 <{thr:>3} px      ->  剔除 {drop:>5}/{total_boxes} 框 ({drop / total_boxes * 100:>4.1f}%)")
    for thr in AREA_RATIO_CANDIDATES:
        drop = sum(1 for a in all_areas if a < thr)
        print(f"    面积 <{thr * 100:>5.2f}%     ->  剔除 {drop:>5}/{total_boxes} 框 ({drop / total_boxes * 100:>4.1f}%)")

    print("\n  规则三 · 剔除过大的框（全图框）：")
    for thr in AREA_MAX_CANDIDATES:
        drop = sum(1 for a in all_areas if a > thr)
        print(f"    面积 >{thr * 100:>5.0f}%     ->  剔除 {drop:>5}/{total_boxes} 框 ({drop / total_boxes * 100:>4.1f}%)")

    print(f"\n【当前阈值下的留存】 {MIN_AREA_RATIO * 100:.2f}%≤面积≤{MAX_AREA_RATIO * 100:.0f}% "
          f"且 短边≥{MIN_SHORT_SIDE_PX}px   同图样本上限 {SAMPLES_PER_IMAGE_CAP}")
    kept_images = 0
    kept_samples = 0
    for _, w, h, rows in images:
        if len(rows) > SANITY_MAX_BOXES:
            continue
        good = [
            r for r in rows
            if short_side_px(r, w, h) >= MIN_SHORT_SIDE_PX
            and MIN_AREA_RATIO <= area_ratio(r) <= MAX_AREA_RATIO
        ]
        if not good:
            continue
        kept_images += 1
        kept_samples += min(len(good), SAMPLES_PER_IMAGE_CAP)

    if not kept_images:
        print("  阈值过严，没有图片留存 —— 请放宽阈值。")
        return

    avg = kept_samples / kept_images
    needed = -(-TARGET_SAMPLES // avg)
    print(f"  保留图片 {kept_images}/{len(images)} ({kept_images / len(images) * 100:.0f}%)")
    print(f"  每图平均产出 {avg:.2f} 条样本")
    print(f"  凑够 {TARGET_SAMPLES:,} 条需要约 {needed:,.0f} 张已标注图片")
    print(f"  （建议按 {needed * 1.25:,.0f} 张备料 —— 多目标样本会一次消耗 2~3 个框，")
    print(f"    另有无法无歧义指代的目标要丢弃，上面的估算没有扣除这两项）")
    print("=" * 68)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, default=BBOX_LABELING_DIR / "coco128" / "labels" / "train2017")
    ap.add_argument("--images-dir", type=Path, default=BBOX_LABELING_DIR / "coco128" / "images" / "train2017")
    ap.add_argument("--classes-yaml", type=Path, default=BBOX_LABELING_DIR / "classes.yaml")
    args = ap.parse_args()

    if not args.labels_dir.exists():
        raise SystemExit(f"找不到标注目录：{args.labels_dir}")

    table = load_class_table(str(args.classes_yaml))
    images = collect(args.labels_dir, args.images_dir, table)
    if not images:
        raise SystemExit(f"{args.labels_dir} 下没有解析出任何标注")
    report(images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

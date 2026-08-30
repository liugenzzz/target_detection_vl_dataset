#!/usr/bin/env python
"""统计标注数据的分布，用于确定质量过滤阈值。

拿到正式数据后【第一件事】就是跑这个。COCO / VisDrone 的阈值不适用于你的数据 ——
目标密度和尺寸分布差别很大，直接套用会误伤。

    python scripts/analyze.py
    python scripts/analyze.py --config other.yaml
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config                      # noqa: E402
from core.cli import _cli  # noqa: E402
from core.classes import load_class_table           # noqa: E402
from core.difficulty import Grader, hard_kept_under_quota   # noqa: E402
from core.yolo import iter_annotations              # noqa: E402

SHORT_SIDE_CANDIDATES = (8, 12, 16, 24)
AREA_MIN_CANDIDATES = (0.0005, 0.001, 0.002, 0.005)
AREA_MAX_CANDIDATES = (0.3, 0.5, 0.7, 0.9)


def pct(sorted_values, p):
    return statistics.quantiles(sorted_values, n=100)[p - 1]


def _wrap(text: str, width: int = 62):
    """按显示宽度折行。中文算两格，直接按字符数折会长短不一。"""
    lines, cur, w = [], "", 0
    for ch in text:
        cw = 2 if ord(ch) > 0x2E80 else 1
        if w + cw > width:
            lines.append(cur)
            cur, w = "", 0
        cur += ch
        w += cw
    if cur:
        lines.append(cur)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    table = load_class_table(cfg.require("paths.classes_yaml"))
    grader = Grader(cfg)

    box_counts, shorts, areas, sizes = [], [], [], []
    grades = Counter()
    label_counts = Counter()

    for ann in iter_annotations(cfg.require("paths.labels_dir"),
                                cfg.require("paths.images_dir"), table,
                                int(cfg.get_path("quality.sanity_max_boxes", 1000))):
        box_counts.append(len(ann.boxes))
        sizes.append((ann.width, ann.height))
        for b in ann.boxes:
            shorts.append(b.short_side_px(ann.width, ann.height))
            areas.append(b.area_ratio)
            label_counts[b.label] += 1
        for g in grader.grade_image(ann.boxes, ann.width, ann.height):
            grades[g.grade] += 1

    if not box_counts:
        raise SystemExit("没有解析出任何标注，检查 paths 配置和类别表")

    shorts.sort(); areas.sort(); box_counts.sort()
    n_img, n_box = len(box_counts), len(areas)
    uniq_sizes = sorted(set(sizes))

    print("=" * 70)
    print(f"图片 {n_img}    标注框 {n_box}    每图均 {n_box / n_img:.2f} 个    类别 {table.count}")
    print(f"分辨率 {len(uniq_sizes)} 种，最小 {min(uniq_sizes)}，最大 {max(uniq_sizes)}")
    print("=" * 70)

    print("\n【每图框数】（参考信息，不作为过滤规则 —— 密集度由难度分级逐目标判断）")
    print(f"  中位 {statistics.median(box_counts):.0f}  p75 {pct(box_counts,75):.0f}  "
          f"p90 {pct(box_counts,90):.0f}  p95 {pct(box_counts,95):.0f}  最大 {max(box_counts)}")

    print("\n【框尺寸】")
    print(f"  短边像素  p5 {pct(shorts,5):.1f}  p10 {pct(shorts,10):.1f}  "
          f"p25 {pct(shorts,25):.1f}  中位 {statistics.median(shorts):.1f}")
    print(f"  面积占比  p5 {pct(areas,5)*100:.3f}%  p10 {pct(areas,10)*100:.3f}%  "
          f"p25 {pct(areas,25)*100:.3f}%  中位 {statistics.median(areas)*100:.3f}%")
    print(f"  等效像素  p5 {grader.equivalent_px(pct(areas,5)):.1f}  "
          f"p25 {grader.equivalent_px(pct(areas,25)):.1f}  "
          f"中位 {grader.equivalent_px(statistics.median(areas)):.1f}   <- 模型实际看到的")

    print("\n  剔除过小目标（候选阈值）：")
    for t in SHORT_SIDE_CANDIDATES:
        d = sum(1 for v in shorts if v < t)
        print(f"    短边 <{t:>3}px    剔除 {d:>6}/{n_box} ({d/n_box*100:>4.1f}%)")
    for t in AREA_MIN_CANDIDATES:
        d = sum(1 for a in areas if a < t)
        print(f"    面积 <{t*100:>5.2f}%   剔除 {d:>6}/{n_box} ({d/n_box*100:>4.1f}%)  "
              f"等效 {grader.equivalent_px(t):.0f}px")

    print("\n  剔除全图框（候选阈值）：")
    for t in AREA_MAX_CANDIDATES:
        d = sum(1 for a in areas if a > t)
        print(f"    面积 >{t*100:>5.0f}%   剔除 {d:>6}/{n_box} ({d/n_box*100:>4.1f}%)")

    print(f"\n【当前配置下的难度分布】（阈值见 config，hard 配额 {grader.hard_quota*100:.0f}%）")
    total = sum(grades.values())
    for k in ("easy", "medium", "hard", "reject"):
        print(f"    {k:>7}  {grades[k]:>6} ({grades[k]/total*100:>5.1f}%)")

    # 【必须把困难配额算进去】。usable 只是「没被质量闸剔除」的框数，不是
    # 最终入库数 —— balance_hard_quota 会把 hard 下采样到 hard_quota 占比，
    # 简单档少时它是全局瓶颈：easy+medium=323、hard=6 万，配额只允许留 36 个
    # hard，最终入库 359 个框。早先这里直接拿 usable 报数，会告诉你「每图 4.16
    # 条」，而真实值是 0.02 条 —— 差 170 倍，且要跑完全量才发现。
    simple = grades["easy"] + grades["medium"]
    hard = grades["hard"]
    quota = grader.hard_quota
    hard_kept = hard_kept_under_quota(simple, hard, quota)
    kept = simple + hard_kept

    cap = int(cfg.get_path("sampling.samples_per_image_cap", 8))
    est = min(kept / n_img, cap) if n_img else 0.0
    print(f"\n  过困难配额后入库 {kept}/{grades['easy'] + grades['medium'] + hard} 个框"
          f"（hard 保留 {hard_kept}/{hard}）")
    if est <= 0:
        print("  预计产出 0 条样本 —— 见下方告警")
    else:
        print(f"  预计每图产出 ~{est:.2f} 条样本，凑 10 万条需约 "
              f"{-(-100000 // est):,.0f} 张图（你现有 {n_img:,} 张）")

    # ---- 告警：配置和这份数据不匹配时，上面的数字会小得离谱 ----
    warns = list(grader.config_conflicts())
    if hard and hard_kept < hard * 0.5:
        warns.append(
            f"困难配额是当前瓶颈：简单档只有 {simple} 个框，按 hard_quota="
            f"{quota:.2f} 只能配 {hard_kept} 个 hard，{hard - hard_kept} 个被丢弃。"
            f"先把简单档做上去（放宽 quality/difficulty 阈值），"
            f"不要直接调大 hard_quota —— 那是把「困难目标只占 10%」这条指标关掉。")
    if n_img and est * n_img < 100000:
        warns.append(
            f"按当前阈值全量跑完约 {est * n_img:,.0f} 条，够不到 10 万条。"
            f"需要过滤后可用框 >= 100000，即 reject 率 <= "
            f"{(1 - 100000 / n_box) * 100:.0f}%（当前 "
            f"{grades['reject'] / n_box * 100:.0f}%）。")
    if warns:
        print("\n【告警】")
        for w in warns:
            print("  ! " + "\n    ".join(_wrap(w)))

    conf = table.confusable_summary()
    if conf:
        print(f"\n【易混类别组】共 {len(conf)} 组（视觉难区分，已在 metadata 打标记）")
        for names in list(conf.values())[:8]:
            present = [n for n in names if n in label_counts]
            if len(present) >= 2:
                print("    " + " / ".join(f"{n}({label_counts[n]})" for n in present))
    print("=" * 70)
    return 0






if __name__ == "__main__":
    raise SystemExit(_cli(main))

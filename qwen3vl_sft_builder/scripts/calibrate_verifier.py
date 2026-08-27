#!/usr/bin/env python
"""标定反向验证器：在开启第 4 步验证之前，必须先跑这个。

反向验证用同一个 VLM 去检验指代表达好不好：把指代连同原图喂回去，让它输出框，
和原框 IoU 达标才保留样本。问题是 —— **验证器自己的定位能力也有限**。
IoU 不达标可能是指代写得不好，也可能是验证器根本框不准。分不清这两者，
阈值就没法定，会大量误杀好样本。

做法：拿【最没有歧义】的指代喂回去，看 IoU 分布。用两档基线：

    上界  「框出图中的{类别}」且该类在图中唯一 —— 这是验证器能力的天花板，
          连这个都低，说明验证器不可用，或者阈值必须往下调很多。
    实测  真实生成的指代骨架 —— 和上界的差距才是指代质量的损失。

    python scripts/calibrate_verifier.py -n 100
"""
from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prompts                                          # noqa: E402
from config import load_config                          # noqa: E402
from core.classes import load_class_table               # noqa: E402
from core.coords import yolo_to_bbox2d                   # noqa: E402
from core.difficulty import REJECT, Grader               # noqa: E402
from core.refer_strategy import ASKABLE, decide_all      # noqa: E402
from core.refer_verify import iou, normalize, parse_box  # noqa: E402
from core.vlm_client import FatalVlmError, VlmClient     # noqa: E402
from core.yolo import iter_annotations                   # noqa: E402


def _pct(vals, p):
    return statistics.quantiles(sorted(vals), n=100)[p - 1] if len(vals) > 1 else (vals[0] if vals else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("-n", type=int, default=100, help="每档各测多少条")
    args = ap.parse_args()

    cfg = load_config(args.config)
    table = load_class_table(cfg.require("paths.classes_yaml"))
    grader = Grader(cfg)
    vlm = VlmClient(cfg)
    if not vlm.enabled:
        raise SystemExit("vlm.enabled 为 false，无法标定验证器")
    scale = int(cfg.get_path("coords.scale", 1000))

    upper, actual = [], []          # (image_path, referring, 真值框)
    for ann in iter_annotations(cfg.require("paths.labels_dir"),
                                cfg.require("paths.images_dir"), table):
        grades = {g.box_index: g for g in grader.grade_image(ann.boxes, ann.width, ann.height)}
        kept = [b for b in ann.boxes if grades[b.index].grade != REJECT]
        if not kept:
            continue
        counts = collections.Counter(b.label for b in kept)
        b2d = lambda b: yolo_to_bbox2d(b.cx, b.cy, b.w, b.h, ann.width, ann.height, scale)

        if len(upper) < args.n:
            for b in kept:
                if counts[b.label] == 1:
                    upper.append((ann.image_path, f"框出图中的{b.label}", b2d(b)))
                    break
        if len(actual) < args.n:
            for idx, sk in decide_all(kept, counts).items():
                if sk.strategy in ASKABLE:
                    box = next(x for x in kept if x.index == idx)
                    actual.append((ann.image_path, sk.phrase, b2d(box)))
                    break
        if len(upper) >= args.n and len(actual) >= args.n:
            break

    for name, batch in (("上界（类别唯一，直接说类别名）", upper),
                        ("实测（真实生成的指代骨架）", actual)):
        if not batch:
            print(f"\n{name}: 没有可测样本")
            continue
        tasks = [(p, gt, "calib", prompts.render("refer_verify", referring=r))
                 for p, r, gt in batch]
        try:
            vlm.prefetch(tasks, progress_every=50)
        except FatalVlmError as exc:
            raise SystemExit(f"\n模型服务配置有问题：\n    {exc}\n")

        scores, unparsed = [], 0
        for (p, referring, gt), (_, key, _, _) in zip(batch, tasks):
            raw = vlm._memory.get(vlm._key(p, key))
            box = parse_box(raw) if raw else None
            if box is None:
                unparsed += 1
                continue
            scores.append(iou(normalize(box, scale), gt))

        if not scores:
            print(f"\n{name}: {len(batch)} 条全部无法解析出框")
            continue
        print(f"\n{name}  共 {len(batch)} 条，解析失败 {unparsed} 条")
        print(f"  IoU  中位 {statistics.median(scores):.3f}   "
              f"p25 {_pct(scores,25):.3f}   p75 {_pct(scores,75):.3f}")
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            keep = sum(1 for s in scores if s >= thr)
            print(f"    阈值 {thr}  ->  保留 {keep}/{len(scores)} ({keep/len(scores)*100:.0f}%)")

    print("\n怎么读这份结果：")
    print("  上界代表验证器的能力天花板。若上界在 0.5 阈值下就保不住多少，")
    print("  说明验证器本身不准，阈值要往下调，或者这一步不该开。")
    print("  实测与上界的差距，才是指代质量真正的损失。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

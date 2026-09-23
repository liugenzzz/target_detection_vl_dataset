#!/usr/bin/env python
"""构建入口。

    python scripts/build.py                    # 用 config/local.yaml
    python scripts/build.py --sample 2000      # 试跑：全库随机抽 2000 张
    python scripts/build.py --config other.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config          # noqa: E402
from core.cli import _cli  # noqa: E402
from core.pipeline import build         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="额外的配置文件，覆盖 default/local")
    ap.add_argument("--limit", type=int,
                    help="只处理【目录里靠前的】前 N 张合格图。文件名带源前缀时"
                         "整批会集中在一个源上，试跑请用 --sample")
    ap.add_argument("--sample", type=int,
                    help="从全部候选里随机抽 N 张再跑（试跑用，各源按占比出现）。抽的是扫描前的张数，过完质量闸剩下的会少一些")
    ap.add_argument("--no-vlm", action="store_true", help="强制关闭 VLM，全部用模板")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.limit and args.sample:
        ap.error("--limit 和 --sample 是两种截法，同时给会互相削：先随机抽 "
                 f"{args.sample} 张，再从中取靠前的 {args.limit} 张，等于又抽了一次。"
                 "试跑用 --sample。")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    if args.no_vlm:
        cfg.setdefault("vlm", {})["enabled"] = False

    stats = build(cfg, limit=args.limit, sample=args.sample)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    from core.tasks import MAIN_LINE, TASKS

    print("\n按任务类型分布（metadata.task_type，可据此筛选）：")
    print("  任务                  实得          配比   差")
    total = stats["samples_total"] or 1
    target = stats.get("task_ratio_target", {})
    surprises = []
    for name in TASKS:
        n = stats["by_task_type"].get(name, 0)
        got = n / total * 100
        want = target.get(name, 0.0) * 100
        mark = " ←主线" if name in MAIN_LINE else ""
        bar = "█" * round(got / 100 * 30)
        # 配比列是【合并后真正生效的】那份。只印实得看不出 local.yaml 没装上。
        cell = f"{want:>5.1f}%" if name in target else "   -- "
        print(f"  {name:<18} {n:>6} {got:>5.1f}%  {cell} "
              f"{got - want:>+5.1f}  {bar}{mark}")
        if name not in target and n:
            surprises.append(f"{name} 配比是 0 却出了 {n} 条")
        elif name in target and not n and want >= 1.0:
            surprises.append(f"{name} 配了 {want:.0f}% 却一条没出")
    print(f"\n  主线合计 {stats['main_line_ratio'] * 100:.1f}%"
          f"    短答案 {stats['short_answer_ratio_actual'] * 100:.1f}%")
    if surprises:
        # 权重 0 的任务在调度器里是直接从 target 里剔掉的，出不来 —— 真出了
        # 就说明跑的不是你以为的那份配置（八成是 local.yaml 没改到）。
        print("\n[警告] 实得和配比对不上，跑的可能不是你以为的那份配置：")
        for line in surprises:
            print(f"  - {line}")

    if stats["task_unavailable"]:
        print("\n因条件不满足而跳过的（该图上出不了这个任务）：")
        for k, v in sorted(stats["task_unavailable"].items(), key=lambda x: -x[1]):
            print(f"  {k:<18} {v:>5} 次")

    split = stats["split"]
    if split["group_overlap"]:
        print(f"\n[警告] train/val 有 {split['group_overlap']} 个来源分组重叠，会造成泄漏！")
    else:
        print(f"\ntrain {split['train']} 条 / val {split['val']} 条，来源分组无重叠 ✓")
    return 0






if __name__ == "__main__":
    raise SystemExit(_cli(main))

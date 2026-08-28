#!/usr/bin/env python
"""构建入口。

    python scripts/build.py                    # 用 config/local.yaml
    python scripts/build.py --limit 200        # 试跑前 200 张图
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
    ap.add_argument("--limit", type=int, help="只处理前 N 张图（试跑用）")
    ap.add_argument("--no-vlm", action="store_true", help="强制关闭 VLM，全部用模板")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    if args.no_vlm:
        cfg.setdefault("vlm", {})["enabled"] = False

    stats = build(cfg, limit=args.limit)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    from core.tasks import MAIN_LINE, TASKS

    print("\n按任务类型分布（metadata.task_type，可据此筛选）：")
    total = stats["samples_total"] or 1
    for name in TASKS:
        n = stats["by_task_type"].get(name, 0)
        mark = " ←主线" if name in MAIN_LINE else ""
        bar = "█" * round(n / total * 40)
        print(f"  {name:<18} {n:>5}  {n / total * 100:>5.1f}%  {bar}{mark}")
    print(f"\n  主线合计 {stats['main_line_ratio'] * 100:.1f}%"
          f"    短答案 {stats['short_answer_ratio_actual'] * 100:.1f}%")

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

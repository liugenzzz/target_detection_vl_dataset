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

    split = stats["split"]
    if split["group_overlap"]:
        print(f"\n[警告] train/val 有 {split['group_overlap']} 个来源分组重叠，会造成泄漏！")
    else:
        print(f"\ntrain {split['train']} 条 / val {split['val']} 条，来源分组无重叠 ✓")
    print(f"困难目标占比 {stats['hard_ratio_in_single'] * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

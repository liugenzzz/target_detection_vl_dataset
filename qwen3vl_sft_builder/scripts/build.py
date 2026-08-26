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


from core.vlm_client import FatalVlmError  # noqa: E402


def _cli(entry):
    """配置类错误直接打印人话并退出，不甩 Python 堆栈 —— 这类错误是用户改配置就能解决的，
    堆栈只会淹没真正有用的那句提示。真正的程序 bug 仍然照常抛出。"""
    import sys
    try:
        return entry()
    except FatalVlmError as exc:
        print(f"\n模型服务配置有问题，已中止（重试也不会成功）：\n\n    {exc}\n\n"
              f"改好后先跑 python scripts/check_vlm.py 确认三步都通过。\n",
              file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError) as exc:
        print(f"\n配置有问题：\n{exc}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断。已完成的 VLM 结果都在缓存里，重跑会从断点继续。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(_cli(main))

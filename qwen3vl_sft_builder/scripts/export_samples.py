#!/usr/bin/env python
"""每个任务抽 N 条，合成一个 jsonl，用来人工过一遍各任务长什么样。

    python scripts/export_samples.py                 # 每任务 10 条
    python scripts/export_samples.py -n 30 --review  # 每任务 30 条，只要质检通过的

产出 samples/preview.jsonl（与 train.jsonl 同格式，可直接喂 LLaMA-Factory 试跑）
和 samples/preview.md（按任务分组的可读版，用来快速扫一眼）。

抽样按任务类型分层，不是随机截断 —— 随机截断会让占比小的任务
（spatial_relation 只有 3%）一条都抽不到。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config                   # noqa: E402
from core.cli import _cli                        # noqa: E402
from core import review                          # noqa: E402
from core.tasks import MAIN_LINE, TASKS          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--jsonl", type=Path, help="默认 <output_dir>/train.jsonl")
    ap.add_argument("-n", type=int, default=10, help="每个任务抽几条")
    ap.add_argument("--review", action="store_true",
                    help="只抽质检通过的（需要先跑 review.py）")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.get_path("paths.output_dir", "./output"))
    src = args.jsonl or (out_dir / "train.jsonl")
    if not src.exists():
        raise SystemExit(f"找不到 {src}，先跑 python scripts/build.py")

    rows = review.load_jsonl(src)
    if args.review:
        rows = [r for r in rows
                if ((r.get("metadata") or {}).get("review") or {}).get("passed")]
        if not rows:
            raise SystemExit("没有质检通过的样本 —— 先跑 review.py，"
                             "或者去掉 --review")

    by_task = defaultdict(list)
    for r in rows:
        by_task[(r.get("metadata") or {}).get("task_type", "?")].append(r)

    rng = random.Random(args.seed)
    picked, short = [], []
    for task in TASKS:                      # 按 TASKS 的顺序，主线在前
        group = by_task.get(task, [])
        if len(group) < args.n:
            short.append(f"{task}({len(group)})")
        picked += rng.sample(group, min(args.n, len(group)))

    dst = args.out_dir or (Path(__file__).resolve().parents[1] / "samples")
    dst.mkdir(parents=True, exist_ok=True)
    n = review.dump_jsonl(dst / "preview.jsonl", picked)
    (dst / "preview.md").write_text(_markdown(picked, src, args.n), encoding="utf-8")

    print(f"{src.name}：{len(rows)} 条 -> 每任务抽 {args.n} 条，共 {n} 条")
    for task in TASKS:
        got = sum(1 for r in picked
                  if (r.get("metadata") or {}).get("task_type") == task)
        mark = "🔷" if task in MAIN_LINE else "  "
        print(f"  {mark} {task:<20} {got:>3} 条  （池子里有 {len(by_task.get(task, []))} 条）")
    if short:
        print(f"\n  不够 {args.n} 条的：{'、'.join(short)} —— 加大 --limit 再跑一次构建")
    print(f"\n  {dst / 'preview.jsonl'}   同 train.jsonl 格式，可直接试跑")
    print(f"  {dst / 'preview.md'}      按任务分组的可读版")
    return 0


def _markdown(picked, src, n) -> str:
    out = [f"# 各任务样本预览（每任务 {n} 条）", "",
           f"来源 `{src.name}`，共 {len(picked)} 条。"
           f"完整数据在同目录的 `preview.jsonl`。", ""]
    out += _provenance(src)
    return "\n".join(out + _task_blocks(picked))


def _provenance(src: Path) -> list:
    """把【哪个模型生成的】写进头部。

    问法同质化、属性跟图对不上，这两件事在桩服务上是必然的、在真模型上才是问题。
    分不清的话看的人会拿桩数据的表现去评判真模型 —— 这已经误导过两次了。
    与其猜，不如直接把模型名写出来：确定、不需要判据。
    """
    report = src.with_name("build_report.json")
    if not report.exists():
        return ["> 找不到 `build_report.json`，无法确认是哪个模型生成的。", ""]
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    eps = data.get("vlm_endpoints") or {}
    calls = data.get("vlm_calls") or {}
    # 用【拿到了多少条结果】判断，不是【发了多少次请求】—— 全部命中缓存时
    # 请求数是 0，但文字确实来自模型，说成「没调用任何模型」是错的。
    got_from_model = (calls.get("prefetched", 0) + calls.get("cache", 0)) > 0
    models = sorted({v.get("model", "?") for v in eps.values()})
    if not got_from_model or not models:
        return [
            "> ⚠️ **这份没有调用任何模型，文字全部来自模板兜底。**",
            "> 主线三个任务需要 VLM 给描述，`vlm.enabled` 为 false 时它们一条都出不来。",
            "",
        ]
    line = f"> **生成用的模型：{'、'.join(models)}**"
    failed = calls.get("failed", 0)
    if failed:
        line += f"　⚠️ 但有 {failed} 次调用失败，这批数据不完整"
    if any(m.lower() in ("fake", "stub", "mock", "test") for m in models):
        return [
            line + "　←　这是桩服务，不是真模型。",
            ">",
            "> 桩服务不读提示词，句式和属性都是硬编码的几种 —— 所以问法看着同质化、",
            "> 属性（颜色等）跟图完全对不上。**这是桩的必然结果，不代表真模型的表现。**",
            "> 它能验的只有「代码通不通、格式对不对、闸拦不拦得住」。",
            ">",
            "> 要看真实质量，接上模型服务重跑：",
            "> `python scripts/build.py --limit 400 && python scripts/export_samples.py`",
            "",
        ]
    return [line, ""]



def _task_blocks(picked):
    by_task = defaultdict(list)
    for r in picked:
        by_task[(r.get("metadata") or {}).get("task_type", "?")].append(r)
    out = []
    for task in TASKS:
        group = by_task.get(task, [])
        if not group:
            continue
        out += [f"## {task}" + ("　🔷主线" if task in MAIN_LINE else ""), ""]
        for i, r in enumerate(group, 1):
            meta = r.get("metadata") or {}
            extra = "　".join(f"{k}={v}" for k, v in meta.items()
                              if k in ("label", "attribute", "attribute_kind",
                                       "relation", "relation_axis", "polarity",
                                       "hard_negative", "n_boxes", "inventory",
                                       "question_source", "answer_format"))
            out.append(f"**{i}.** `{meta.get('source_image', '')}`　{extra}")
            out.append("")
            for t in r.get("conversations", []):
                who = "**问**" if t.get("from") == "human" else "**答**"
                val = str(t.get("value", "")).replace("<image>\n", "")
                out.append(f"- {who} {val}")
            out.append("")
    return out


if __name__ == "__main__":
    raise SystemExit(_cli(main))

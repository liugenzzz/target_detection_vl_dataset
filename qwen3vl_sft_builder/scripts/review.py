#!/usr/bin/env python
"""全量质检：每条问答对让大模型对着原图核对一遍并打分。

    python scripts/review.py                                  # 审 output/train.jsonl
    python scripts/review.py --jsonl output/val.jsonl
    python scripts/review.py --limit 200                      # 先小批量看看分布
    python scripts/review.py --min-score 4                    # 收紧阈值重判
    python scripts/review.py --dry-run                        # 只用已有缓存重判，不发请求

构建阶段的过滤全是结构性的（占位符、类别泄漏、描述长度、坐标越界），都不看图。
框偏了、参照物是编的、指代不唯一 —— 这些只有看图才发现得了。

产出（与 --jsonl 同目录）：

    train.reviewed.jsonl   通过的，带 metadata.review 评分
    train.rejected.jsonl   没通过的，带 review.reason，人工看一眼再决定
    review_report.json     分数分布、各维度均值、按任务的通过率、驳回原因 top

**不直接删**：审核模型自己也会看错，把它的判断当成建议而不是判决。

成本：按图分组，一张图上的样本合并成一次调用。十万条样本按每图 8 条算，
约一万两千次调用。结果按图落盘缓存，中断了重跑接着走。

**别用生成它的同一个模型自审** —— 它会倾向于认同自己的输出。
config 里 vlm.roles.review 可以给审核单独指定模型或整个模型池。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prompts                                       # noqa: E402
from config import load_config                       # noqa: E402
from core import review                              # noqa: E402
from core.cli import _cli                         # noqa: E402
from core.vlm_client import VlmClient             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--jsonl", type=Path, help="默认 <output_dir>/train.jsonl")
    ap.add_argument("--limit", type=int, help="只审前 N 张图（试跑用）")
    ap.add_argument("--min-score", type=int, help="覆盖 review.min_score")
    ap.add_argument("--dry-run", action="store_true",
                    help="不发请求，只用已有缓存重新判定（改阈值后重跑用）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.get_path("paths.output_dir", "./output"))
    src = args.jsonl or (out_dir / "train.jsonl")
    if not src.exists():
        raise SystemExit(f"找不到 {src}，先跑 python scripts/build.py")

    min_score = args.min_score or int(cfg.get_path("review.min_score", 3))
    min_dim = dict(cfg.get_path("review.min_dimension", {}) or {})
    max_per_call = int(cfg.get_path("review.max_samples_per_call", 8))

    images_dir = Path(cfg.get_path("paths.images_dir", "") or ".")
    samples = review.load_jsonl(src)
    groups = review.group_by_image(samples)
    print(f"{src.name}：{len(samples)} 条样本，{len(groups)} 张图")

    # images 字段默认存的是裸文件名（output.image_path_style: filename），
    # 质检要读图，得先拼回磁盘上的真实路径。
    resolved = {img: review.resolve_image(img, images_dir) for img in groups}
    missing = [i for i, p in resolved.items() if p is None]
    if missing:
        print(f"  {len(missing)} 张图找不到（如 {missing[0]}），这些样本无法质检。\n"
              f"  检查 paths.images_dir 是否指向 {src.name} 所用的那批图。")
        if len(missing) == len(groups):
            raise SystemExit("一张图都找不到，先把 paths.images_dir 配对")

    # 一张图上的样本太多就切块 —— 一次让模型判二十条，它会开始敷衍
    chunks = []
    for image, group in groups.items():
        if resolved.get(image) is None:
            continue
        for i in range(0, len(group), max_per_call):
            chunks.append((resolved[image], group[i:i + max_per_call]))
    if args.limit:
        chunks = chunks[:args.limit]

    client = VlmClient(cfg, role="review")
    if not client.enabled and not args.dry_run:
        raise SystemExit("vlm.enabled 为 false，无法质检")
    print(f"审核模型：{'、'.join(e.model for e in client.endpoints)}"
          f"（{len(chunks)} 次调用，并发 {client.concurrency}）")
    if not client.cache_dir and not args.dry_run:
        print("  [注意] vlm.cache_dir 没配，审核结果不落盘。"
              f"这 {len(chunks)} 次调用中断一次就要全部重来，"
              "\n         而且改阈值重判也得重新调用一遍。全量跑之前先把它配上。")
    if len(client.endpoints) == 1 and not (cfg.get_path("vlm.roles.review") or {}):
        print("  [注意] 质检用的是生成数据的同一个模型，它会倾向于认同自己的输出。"
              "\n         有第二个模型时在 vlm.roles.review 里指过去。")

    # 预取：按 (图, 这一块的样本数) 做缓存键，中断重跑接着走
    if not args.dry_run:
        tasks = []
        for image, group in chunks:
            ann = group[0].get("metadata") or {}
            w = int(ann.get("image_width") or 0) or 1000
            h = int(ann.get("image_height") or 0) or 1000
            text = prompts.render("review", width=w, height=h,
                                  samples=review.render_samples(group))
            tasks.append((image, _key_of(group), "review", text))
        client.prefetch(tasks, label="质检")

    scored = unscored = 0
    for image, group in chunks:
        raw = client.raw_result(image, _key_of(group))
        parsed = review.parse(raw, len(group)) if raw else None
        for i, s in enumerate(group):
            got = (parsed or {}).get(i)
            if not got:
                unscored += 1
                continue
            passed, reason = review.verdict(got, min_score, min_dim)
            s.setdefault("metadata", {})["review"] = {**got, "passed": passed,
                                                      "reason": reason}
            s["review"] = s["metadata"]["review"]      # summarize 取用，落盘前删掉
            scored += 1

    summary = review.summarize(samples)
    for s in samples:
        s.pop("review", None)

    passed = [s for s in samples
              if ((s.get("metadata") or {}).get("review") or {}).get("passed", True)]
    rejected = [s for s in samples
                if ((s.get("metadata") or {}).get("review") or {}).get("passed") is False]

    stem = src.stem
    n_ok = review.dump_jsonl(src.with_name(f"{stem}.reviewed.jsonl"), passed)
    n_bad = review.dump_jsonl(src.with_name(f"{stem}.rejected.jsonl"), rejected)
    report = src.with_name("review_report.json")
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary(summary, scored, unscored, n_ok, n_bad, src, stem)
    return 0


def _key_of(group) -> list:
    """缓存键：这一块里每条样本的第一句问话的哈希。样本变了就重新审。"""
    import hashlib
    h = hashlib.sha1()
    for s in group:
        for t in s.get("conversations", []):
            h.update(str(t.get("value", "")).encode("utf-8"))
    return [int(h.hexdigest()[:8], 16)]


def _print_summary(summary, scored, unscored, n_ok, n_bad, src, stem) -> None:
    if not summary.get("scored"):
        print("\n一条都没judge成功 —— 检查审核模型是否返回了合法 JSON。")
        return
    print(f"\n{'=' * 62}\n质检结果\n{'=' * 62}")
    print(f"  评上分的 {summary['scored']} 条，没判上的 {summary['unscored']} 条")
    print(f"  通过 {summary['passed']}（{summary['pass_rate']:.1%}），"
          f"驳回 {summary['rejected']}")
    print(f"  平均分 {summary['score_avg']}")
    print("\n  分数分布：")
    for k, v in summary["score_dist"].items():
        bar = "█" * max(1, round(v / max(summary["score_dist"].values()) * 34))
        print(f"    {k} 分  {v:5d}  {bar}")
    print("\n  各维度均值：")
    names = {"correct": "答案与图相符", "grounded": "描述没有编造",
             "clear": "指代唯一/描述具体", "instruction": "问句是指令口吻",
             "needs_image": "必须看图才能答"}
    for dim, avg in summary["dimension_avg"].items():
        print(f"    {names.get(dim, dim):16s} {avg}")
    print("\n  按任务：")
    for task, v in summary["by_task"].items():
        print(f"    {task:20s} {v['n']:5d} 条  均分 {v['avg']}  "
              f"通过 {v['pass_rate']:.1%}")
    if summary["top_reject_reasons"]:
        print("\n  驳回原因：")
        for reason, n in summary["top_reject_reasons"].items():
            print(f"    {reason:32s} {n}")
    print(f"\n  通过的 -> {src.with_name(f'{stem}.reviewed.jsonl').name}（{n_ok} 条）")
    print(f"  驳回的 -> {src.with_name(f'{stem}.rejected.jsonl').name}（{n_bad} 条）"
          "  ← 人工扫一眼再决定，审核模型自己也会看错")


if __name__ == "__main__":
    raise SystemExit(_cli(main))

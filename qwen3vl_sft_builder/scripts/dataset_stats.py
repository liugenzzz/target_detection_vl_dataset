#!/usr/bin/env python
"""数据集体检：不调模型、纯离线算的客观指标，跑一次几秒钟。

    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --jsonl output/train.reviewed.jsonl

和 scripts/review.py 的分工：

    review.py         【主观】让大模型看图逐条打分。能发现「框住的是旁边那棵树」
    dataset_stats.py  【客观】离线统计分布。能发现「347 类只覆盖了 40 类」
                      「85% 的框挤在画面中央」「描述里提到了标注文件里没有的类别」

后者发现的都是【分布层面】的问题 —— 单看任何一条样本都是好的，只有汇总才看得出来。
主观质检逐条打分永远发现不了。

指标对标公开做法：CHAIR（幻觉率）、Distinct-n（词汇多样性）、POPE（存在性
问答的正负平衡与负采样难度）、基尼系数（长尾）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config                   # noqa: E402
from core.cli import _cli                        # noqa: E402
from core import describe_kinds, review, stats   # noqa: E402
from core.classes import load_class_table        # noqa: E402
import prompts                                  # noqa: E402
from core.yolo import iter_annotations           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--jsonl", type=Path, help="默认 <output_dir>/train.jsonl")
    ap.add_argument("--out", type=Path, help="写 json，默认与 jsonl 同目录")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.get_path("paths.output_dir", "./output"))
    src = args.jsonl or (out_dir / "train.jsonl")
    if not src.exists():
        raise SystemExit(f"找不到 {src}，先跑 python scripts/build.py")

    table = load_class_table(cfg.require("paths.classes_yaml"))
    samples = review.load_jsonl(src)

    # 每张图标注里真实出现过的类别 —— CHAIR 的真值
    truth = defaultdict(set)
    for ann in iter_annotations(Path(cfg.require("paths.labels_dir")),
                                Path(cfg.require("paths.images_dir")), table,
                                int(cfg.get_path("quality.sanity_max_boxes", 1000))):
        truth[ann.image_path.name] |= {b.label for b in ann.boxes}

    rep = stats.report(samples, dict(truth), sorted(table.id2name.values()),
                       int(cfg.get_path("coords.scale", 1000)),
                       list(describe_kinds.load_all()))
    dst = args.out or src.with_name("dataset_stats.json")
    dst.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(rep, src, dst)
    return 0


def _bar(v, width=26):
    return "█" * max(0, round(v * width))


def _print(r, src, dst) -> None:
    print(f"\n{'=' * 66}\n{src.name} 体检报告 —— {r['samples']} 条样本\n{'=' * 66}")

    h = r["hallucination_chair"]
    print(f"\n【幻觉率】CHAIR —— 答案里提到的类别有多少不在该图标注里")
    print(f"  CHAIR_i  {h['chair_i']:.2%}   提到 {h['mentions_total']} 次，"
          f"说错 {h['mentions_wrong']} 次")
    print(f"  CHAIR_s  {h['chair_s']:.2%}   有这么多条样本至少说错一个")
    if h["top_hallucinated"]:
        print(f"  最常说错：{h['top_hallucinated']}")
    print("  注：只统计类别表里的词。描述里的「斑马线」「路灯杆」不在表中，")
    print("      无从判定真假，不计入 —— 这会低估真实幻觉率，但绝不误报。")

    c = r["class_coverage"]
    print(f"\n【类别覆盖】{c['classes_covered']}/{c['classes_in_table']} 类"
          f"（{c['coverage_rate']:.1%}） {_bar(c['coverage_rate'])}")
    print(f"  基尼系数 {c['gini']}（0=每类一样多，1=全挤在一类上）")
    print(f"  样本数不足 10 条的类别：{c['tail_lt_10']} 个")
    print(f"  最多的五类：{c['top5']}")
    if c["never_used"]:
        print(f"  一条都没有的（前 20）：{c['never_used']}")

    b = r["box_distribution"]
    print(f"\n【框的分布】共 {b['boxes_total']} 个框")
    for zone, share in list(b["zone_share"].items())[:9]:
        print(f"    {zone}  {share:>6.1%}  {_bar(share * 3)}")
    print(f"  最挤的一格占 {b['max_zone_share']:.1%}（九宫格均匀是 11.1%，"
          f"超过 25% 就该看看数据源）")
    print(f"  框面积占比  p10 {b['area_p10']:.4%}  p50 {b['area_p50']:.4%}  "
          f"p90 {b['area_p90']:.4%}")

    a = r["answer_shape"]
    print(f"\n【答案多样性】{a['text_answers']} 条文字答案"
          f"（框答案不计）")
    print(f"  完全不重样 {a['distinct_rate']:.1%}   "
          f"Distinct-2 {a['distinct_2']}   Distinct-3 {a['distinct_3']}")
    print(f"  长度  p10 {a['len_p10']}  p50 {a['len_p50']}  p90 {a['len_p90']} 字")
    if a["top_openings"]:
        print("  开头四个字最集中的：")
        for k, v in a["top_openings"].items():
            print(f"    {k!r:12s} {v:>6.1%}  {_bar(v * 3)}")
        n = a.get("opening_kinds") or 1
        mark = "✓" if a.get("opening_ok") else "✗"
        print(f"    {mark} 描述分 {n} 种子类型，答案结构不同、开头自然也该散开；"
              f"最集中的一种占 {a.get('opening_max_share', 0):.1%}"
              f"（均匀约 {1 / n:.1%}）")
        if not a.get("opening_ok"):
            print("      超过均值两倍 —— 七种正在退化成一种。")
            print("      看下面各子类型的答案长度：哪一种和 full 差不多长，")
            print("      就是它的 prompts/describe/<子类型>.txt 要求写得还不够硬。")

    mix = r.get("describe_kind_mix") or {}
    if mix:
        print("\n【描述子类型】各出了多少、答案多长")
        for k, v in mix.items():
            print(f"    {k:<12} {v['n']:>5} 条  {v['share']:>6.1%}  平均 {v['len_avg']} 字")
        print("    某一种的长度和 full 差不多，说明它没照着自己的要求写")

    e = r["exist_balance"]
    if e:
        print(f"\n【存在性问答】POPE 口径 —— {e['total']} 条")
        print(f"  答「有」{e['positive_rate']:.1%}（失衡会让模型学成一律答有或一律答没有）")
        print(f"  难负样本 {e['hard_negative_rate']:.1%}（问的是易混类别，"
              f"不是不相干的东西 —— 后者答「没有」不用看图）")

    print(f"\n已写入 {dst}")


if __name__ == "__main__":
    raise SystemExit(_cli(main))

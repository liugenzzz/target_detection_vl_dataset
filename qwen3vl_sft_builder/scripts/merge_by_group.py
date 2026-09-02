#!/usr/bin/env python3
"""把补跑的样本并进已有划分 —— 按【来源分组】归位，不重新划分。

    python scripts/merge_by_group.py --into output --add output_inv [更多目录 ...]

【为什么不能让补跑自己划分】_split_by_source 是按样本总数和 shuffle 后的组顺序
填 test/val/train 的。补跑一次样本集不同 -> 每个来源组的落位就不同 ->
第一批 test 里的某个来源组，可能在补跑里落进 train。同一张原图跨了划分，
评估数字就废了，而且从结果上看不出来。

正确做法：已有划分是【基准】，补跑的每条样本按它的来源组去查基准属于哪一份，
放进对应文件。基准里没见过的组（补跑覆盖了新图片）按配置比例补分，并单独报数。

【去重】detect_describe 是 detect_class 的超集：第一轮的问句和答案完全一样，
只是多一轮坐标回指的描述。同一个 (图, 类别) 上两者都存在时，保留
detect_describe、丢弃 detect_class —— 否则合并后会有一批第一轮一字不差的
近重复样本。混跑时 used 机制会挡住这种撞车，但单独跑时 used 是空的。

原地改写 --into 目录的三个 jsonl，改前自动备份为 *.jsonl.bak。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.grouping import source_group_key          # noqa: E402

SPLITS = ("train", "val", "test")

# 超集任务 -> 被它顶掉的任务。键任务覆盖了同一个 (图, 类别) 时，值任务那条丢弃。
SUPERSEDES = {"detect_describe": "detect_class"}


def _read(path: Path) -> list:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write(path: Path, rows: list) -> None:
    # newline="\n"：Windows 上文本模式会把 \n 写成 \r\n，同一份数据在不同机器上
    # 跑出来的字节不一致，校验和对不上。
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _group_of(sample: dict) -> str:
    return source_group_key(sample["images"][0])


def _task_of(sample: dict) -> str:
    return (sample.get("metadata") or {}).get("task_type", "")


def _coverage_key(sample: dict):
    """(图, 类别) —— 判断两条样本是不是覆盖同一个目标集合。"""
    label = (sample.get("metadata") or {}).get("label")
    return (sample["images"][0], label) if label else None


def _drop_superseded(base: dict, incoming: list) -> Counter:
    """把被超集任务顶掉的旧样本从 base 里删掉，返回按任务计数。"""
    covered = {}
    for row in incoming:
        sup = SUPERSEDES.get(_task_of(row))
        key = _coverage_key(row)
        if sup and key:
            covered.setdefault(sup, set()).add(key)
    if not covered:
        return Counter()
    dropped = Counter()
    for split in SPLITS:
        keep = []
        for row in base[split]:
            task, key = _task_of(row), _coverage_key(row)
            if key and key in covered.get(task, ()):
                dropped[task] += 1
                continue
            keep.append(row)
        base[split] = keep
    return dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--into", required=True, help="基准目录，含 train/val/test.jsonl")
    ap.add_argument("--add", nargs="+", required=True, help="要并入的目录（可多个）")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--test-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--dry-run", action="store_true", help="只报数不写文件")
    args = ap.parse_args()

    base_dir = Path(args.into)
    base = {s: _read(base_dir / f"{s}.jsonl") for s in SPLITS}
    if not any(base.values()):
        print(f"[失败] {base_dir} 里没有 train/val/test.jsonl")
        return 1

    # 基准划分：来源组 -> 属于哪一份
    home = {}
    for split, rows in base.items():
        for row in rows:
            home.setdefault(_group_of(row), split)
    print(f"基准 {base_dir}：" + " / ".join(f"{s} {len(base[s]):,}" for s in SPLITS)
          + f"   来源组 {len(home):,} 个")

    seen_ids = {row.get("id") for rows in base.values() for row in rows}

    incoming = []
    for d in args.add:
        got = [r for s in SPLITS for r in _read(Path(d) / f"{s}.jsonl")]
        print(f"并入 {d}：{len(got):,} 条")
        incoming.extend(got)

    # 先顶掉被超集任务覆盖的旧样本，再归位新样本 ——
    # 顺序反过来的话，新样本会先进 base，然后被自己顶掉。
    dropped = _drop_superseded(base, incoming)
    for task, n in dropped.items():
        sup = [k for k, v in SUPERSEDES.items() if v == task][0]
        print(f"去重：{sup} 顶掉 {task} {n:,} 条（同一 图×类别）")

    placed = Counter()
    new_groups = {}
    dup = 0
    for row in incoming:
        if row.get("id") in seen_ids:
            dup += 1
            continue
        seen_ids.add(row.get("id"))
        key = _group_of(row)
        if key in home:
            base[home[key]].append(row)
            placed[home[key]] += 1
        else:
            new_groups.setdefault(key, []).append(row)

    # 基准里没见过的来源组：按比例补分，整组不拆
    n_new_rows = sum(len(v) for v in new_groups.values())
    if new_groups:
        keys = sorted(new_groups)
        random.Random(args.seed).shuffle(keys)
        n_test = n_val = 0
        for k in keys:
            rows = new_groups[k]
            if n_test < n_new_rows * args.test_ratio:
                split, n_test = "test", n_test + len(rows)
            elif n_val < n_new_rows * args.val_ratio:
                split, n_val = "val", n_val + len(rows)
            else:
                split = "train"
            base[split].extend(rows)

    print(f"\n按来源组归位 {sum(placed.values()):,} 条"
          f"   新来源组 {len(new_groups):,} 个 / {n_new_rows:,} 条按比例补分")
    if dup:
        print(f"跳过重复 id {dup:,} 条")

    # 【必须为 0】同一个来源组只能出现在一份里
    where = {}
    for split, rows in base.items():
        for row in rows:
            where.setdefault(_group_of(row), set()).add(split)
    leaked = {k: sorted(v) for k, v in where.items() if len(v) > 1}
    print("\n" + " / ".join(f"{s} {len(base[s]):,}" for s in SPLITS)
          + f"   合计 {sum(len(v) for v in base.values()):,} 条")
    if leaked:
        print(f"[失败] {len(leaked)} 个来源组跨了划分，没有写入。示例：")
        for k, v in list(leaked.items())[:5]:
            print(f"    {k} -> {v}")
        return 1
    print("来源分组无重叠 ✓")

    if args.dry_run:
        print("\n--dry-run，未写文件")
        return 0
    for split in SPLITS:
        path = base_dir / f"{split}.jsonl"
        if path.exists():
            shutil.copy2(path, path.with_suffix(".jsonl.bak"))
        _write(path, base[split])
    print(f"\n已写入 {base_dir}（原文件备份为 *.jsonl.bak）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

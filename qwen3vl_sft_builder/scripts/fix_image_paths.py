#!/usr/bin/env python3
"""改写 jsonl 里 images 字段的图片路径。不动其它任何字段。

    # 看一眼会改成什么样，不落盘
    python scripts/fix_image_paths.py --dry-run \
        --images-dir /mnt/data3/code/data_process/jsmb_data_0901/images \
        /mnt/data3/code/trainspace/datasets/mbjc_vqa/*.jsonl

    # 真改（自动留 .bak）
    python scripts/fix_image_paths.py \
        --images-dir /mnt/data3/code/data_process/jsmb_data_0901/images \
        /mnt/data3/code/trainspace/datasets/mbjc_vqa/*.jsonl

    # 反过来，绝对路径改回裸文件名（交付时用）
    python scripts/fix_image_paths.py --to filename \
        --images-dir /mnt/data3/code/data_process/jsmb_data_0901/images \
        train.jsonl val.jsonl test.jsonl

【只认文件名】不管原来写的是裸文件名、绝对路径还是相对路径，一律取
basename 再拼 --images-dir。所以重复跑没有副作用，跑错了再跑一遍也能救回来。

【落盘前先验图在不在】这才是这个脚本存在的理由。images_dir 传错一个字母，
写出去的十万条路径全是坏的，而 jsonl 本身看不出任何异常 —— 要等训练跑起来
才炸。所以有一张图找不到就拒绝写，除非显式 --allow-missing。

【原子写 + 备份】先写 .tmp 再 os.replace，中途挂掉不会留下半截文件；
原文件同时留一份 .bak（--no-backup 关掉）。几百 MB 的 jsonl 重跑一次很贵。

metadata.source_image 不动 —— 它按设计就存裸文件名，是 images 字段被改乱之后
追回原图的唯一线索。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def index_images(images_dir: Path) -> Set[str]:
    """图片目录里的文件名集合。一次列目录，比十万次 stat 快得多。"""
    return {p.name for p in images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS}


def convert(path: Path, images_dir: Path, have: Set[str], to: str,
            dry_run: bool, backup: bool, allow_missing: bool) -> Dict[str, int]:
    stat = {"rows": 0, "changed": 0, "missing": 0, "no_images": 0}
    missing_examples: List[str] = []
    out_lines: List[str] = []

    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [失败] {path.name}:{lineno} 不是合法 JSON：{exc}")
                return {**stat, "fatal": 1}
            stat["rows"] += 1

            images = row.get("images")
            if not images:
                stat["no_images"] += 1
                out_lines.append(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            new = []
            for item in images:
                name = Path(str(item)).name
                if name not in have:
                    stat["missing"] += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(f"{path.name}:{lineno} {name}")
                new.append(name if to == "filename" else str(images_dir / name))
            if new != images:
                stat["changed"] += 1
                row["images"] = new
            out_lines.append(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  {path}")
    print(f"    共 {stat['rows']:,} 条   需改写 {stat['changed']:,} 条"
          + (f"   没有 images 字段 {stat['no_images']:,} 条" if stat["no_images"] else ""))
    if stat["missing"]:
        print(f"    [警告] {stat['missing']:,} 处引用的图片在 {images_dir} 里找不到：")
        for e in missing_examples:
            print(f"      {e}")
        if not allow_missing:
            print("    未落盘。确认 --images-dir 传对了没有；"
                  "确实有图缺失且可以接受，就加 --allow-missing。")
            return {**stat, "fatal": 1}

    if dry_run:
        return stat
    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():                      # 重复跑不要把备份也覆盖掉
            bak.write_bytes(path.read_bytes())
            print(f"    备份 -> {bak.name}")
        else:
            print(f"    备份已存在，跳过：{bak.name}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(out_lines)
    os.replace(tmp, path)                          # 原子替换
    print(f"    已写入")
    return stat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+", type=Path, help="要改的 jsonl，可多个")
    ap.add_argument("--images-dir", required=True, type=Path, help="图片所在目录")
    ap.add_argument("--to", choices=("absolute", "filename"), default="absolute",
                    help="改成绝对路径（默认）还是裸文件名")
    ap.add_argument("--dry-run", action="store_true", help="只报数不写文件")
    ap.add_argument("--no-backup", action="store_true", help="不留 .bak")
    ap.add_argument("--allow-missing", action="store_true",
                    help="有图找不到也照写。默认拒绝 —— 多半是目录传错了")
    args = ap.parse_args()

    images_dir = args.images_dir.resolve()
    if not images_dir.is_dir():
        print(f"[失败] 图片目录不存在：{images_dir}")
        return 1
    have = index_images(images_dir)
    if not have:
        print(f"[失败] {images_dir} 里没有图片")
        return 1
    print(f"图片目录 {images_dir}")
    print(f"图片 {len(have):,} 张   目标格式：{args.to}\n")

    targets = [p for p in args.jsonl if p.suffix == ".jsonl"]
    for p in args.jsonl:
        if p.suffix != ".jsonl":
            print(f"[跳过] 不是 .jsonl：{p}")
    if not targets:
        print("[失败] 没有可处理的 jsonl")
        return 1

    bad = 0
    total = {"rows": 0, "changed": 0, "missing": 0}
    for p in targets:
        if not p.exists():
            print(f"  [跳过] 文件不存在：{p}")
            continue
        st = convert(p, images_dir, have, args.to,
                     args.dry_run, not args.no_backup, args.allow_missing)
        bad += st.get("fatal", 0)
        for k in total:
            total[k] += st.get(k, 0)
        print()

    print(f"合计 {total['rows']:,} 条，改写 {total['changed']:,} 条，"
          f"缺图 {total['missing']:,} 处")
    if args.dry_run:
        print("--dry-run，未改动任何文件")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

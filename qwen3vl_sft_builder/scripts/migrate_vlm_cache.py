#!/usr/bin/env python3
"""把已有的 VLM 缓存改名到当前的提示词指纹下，避免白跑一遍模型。

    python scripts/migrate_vlm_cache.py --old-fp cec787269e39 --dry-run
    python scripts/migrate_vlm_cache.py --old-fp cec787269e39

【为什么需要这个】缓存键是 md5(图片名|bbox|提示词指纹)。指纹一变，全部缓存
就失效 —— 而缓存文件里只存了描述文本，没存图片名，光看文件名反查不出来。

但图片名我们有（images_dir 里就是），bbox 对构建期那次「挑对象」调用固定为
空列表，所以两个键都能重算：按旧指纹算出老文件名，按当前指纹算出新文件名，
改名即可。一次调用都不用重发。

--old-fp 传缓存【写入时】用的那个指纹。不确定的话先 --dry-run，脚本会报出
命中多少个文件；命中数接近图片总数就说明传对了。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config                       # noqa: E402
from core.vlm_client import _prompt_fingerprint      # noqa: E402
from core.yolo import IMAGE_EXTS                     # noqa: E402


def key_of(image_name: str, fingerprint: str) -> str:
    """和 VlmClient._key 保持一致。改那边就要改这里。"""
    return hashlib.md5(f"{image_name}|[]|{fingerprint}".encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--old-fp", required=True, help="缓存写入时用的提示词指纹")
    ap.add_argument("--cache-dir", help="默认取 config 的 vlm.cache_dir")
    ap.add_argument("--images-dir", help="默认取 config 的 paths.images_dir")
    ap.add_argument("--dry-run", action="store_true", help="只报数不改名")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_dir = Path(args.cache_dir or cfg.get_path("vlm.cache_dir", "./output/vlm_cache"))
    images_dir = Path(args.images_dir or cfg.require("paths.images_dir"))
    new_fp = _prompt_fingerprint()

    if not cache_dir.is_dir():
        print(f"[失败] 缓存目录不存在：{cache_dir}")
        return 1
    if args.old_fp == new_fp:
        print(f"旧指纹和当前指纹相同（{new_fp}），无需迁移")
        return 0

    images = [p.name for p in sorted(images_dir.iterdir())
              if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        print(f"[失败] {images_dir} 里没有图片")
        return 1

    print(f"缓存目录 {cache_dir}")
    print(f"图片 {len(images):,} 张   {args.old_fp} -> {new_fp}\n")

    hit = renamed = already = collision = 0
    for name in images:
        src = cache_dir / f"{key_of(name, args.old_fp)}.json"
        if not src.exists():
            continue
        hit += 1
        dst = cache_dir / f"{key_of(name, new_fp)}.json"
        if dst.exists():
            already += 1
            continue
        if args.dry_run:
            renamed += 1
            continue
        try:
            src.rename(dst)
            renamed += 1
        except OSError as exc:
            print(f"  改名失败 {src.name}：{exc}")
            collision += 1

    total = len(list(cache_dir.glob("*.json")))
    print(f"按旧指纹命中 {hit:,} / {len(images):,} 张图"
          f"（占缓存目录 {total:,} 个文件的 {hit / max(total, 1) * 100:.0f}%）")
    print(f"{'待改名' if args.dry_run else '已改名'} {renamed:,}   目标已存在 {already:,}   失败 {collision:,}")
    if hit == 0:
        print("\n[注意] 一个都没命中，--old-fp 多半传错了。")
        print("       缓存目录里的文件名是 md5(图片名|[]|指纹)，指纹不对就全对不上。")
        return 1
    if args.dry_run:
        print("\n--dry-run，未改动任何文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

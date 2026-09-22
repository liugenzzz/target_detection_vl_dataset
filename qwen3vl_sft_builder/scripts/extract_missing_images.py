#!/usr/bin/env python3
"""把清单里有、但 images_dir 里缺的图片，从原始归档里捞出来补齐。

    python scripts/extract_missing_images.py --dry-run
    python scripts/extract_missing_images.py

【什么时候用】选片流程把标注落盘了、图片没落盘（某个源的抽取步骤漏跑或中途
断了）。实测踩到：110,150 个标注对 105,299 张图，差的 4,851 张全是同一个源
（fitrs），归档还在，只是没解出来。

清单里每行都带原始归档的位置：

    "image_locator": {"archive": "/mnt/.../FIT-RS_Img.tar.gz",
                      "member":  "imgv2_split_512_100_vaild/0505__512__2884___5768.png"}
    "selected_image": "/mnt/.../images/fitrs_47575b0247b4fd3ae258.png"

member 是归档里的路径，selected_image 的文件名是落盘之后【重命名过】的名字 ——
两者对不上是正常的，这个脚本负责把它们接起来。

【一个归档只读一遍】。tar.gz 是顺序介质，按成员随机取要把整个流重放一遍，
几千个成员就是几千遍。这里先按归档把要捞的成员归好组，再流式过一遍，
边过边挑。zip 支持随机读，但一并走同一条路，少一个分支。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config                    # noqa: E402
from core.yolo import IMAGE_EXTS                  # noqa: E402


def _have(images_dir: Path) -> set:
    return {p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}


def _plan(manifest: Path, have: set) -> Dict[str, Dict[str, str]]:
    """{归档路径: {归档内成员路径: 要落盘的文件名}}。"""
    todo: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = row.get("selected_image") or ""
        if not target:
            continue
        name = Path(str(target)).name
        if Path(name).stem in have:
            continue
        loc = row.get("image_locator") or {}
        archive, member = loc.get("archive"), loc.get("member")
        if archive and member:
            todo[str(archive)][str(member)] = name
    return todo


def _pull_tar(archive: Path, want: Dict[str, str], out: Path, dry: bool) -> int:
    done = 0
    mode = "r|gz" if archive.suffix in (".gz", ".tgz") else "r|"
    with tarfile.open(archive, mode) as tf:
        for m in tf:
            name = want.get(m.name)
            if not name or not m.isfile():
                continue
            if not dry:
                src = tf.extractfile(m)
                if src is None:
                    continue
                (out / name).write_bytes(src.read())
            done += 1
            if done == len(want):
                break            # 要的都拿到了，不必把剩下的流读完
    return done


def _pull_zip(archive: Path, want: Dict[str, str], out: Path, dry: bool) -> int:
    done = 0
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for member, name in want.items():
            if member not in names:
                continue
            if not dry:
                (out / name).write_bytes(zf.read(member))
            done += 1
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--manifest", type=Path, help="默认取 paths.selection_manifest")
    ap.add_argument("--images-dir", type=Path, help="默认取 paths.images_dir")
    ap.add_argument("--dry-run", action="store_true", help="只报数不解压")
    args = ap.parse_args()

    cfg = load_config(args.config)
    images_dir = Path(args.images_dir or cfg.require("paths.images_dir"))
    manifest = Path(args.manifest or cfg.require("paths.selection_manifest"))
    if not images_dir.is_dir():
        print(f"[失败] 图片目录不存在：{images_dir}")
        return 1

    have = _have(images_dir)
    todo = _plan(manifest, have)
    n_missing = sum(len(v) for v in todo.values())
    print(f"图片目录 {images_dir}   已有 {len(have):,} 张")
    if not n_missing:
        print("清单里的图片都在，没有要补的")
        return 0

    print(f"缺 {n_missing:,} 张，分布在 {len(todo)} 个归档：\n")
    for a, want in sorted(todo.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(want):>7,}  {a}" + ("" if Path(a).exists() else "   [归档不存在]"))
    print()

    total = 0
    for a, want in todo.items():
        archive = Path(a)
        if not archive.exists():
            print(f"[跳过] 归档不存在：{archive}")
            continue
        print(f"读 {archive.name}（要捞 {len(want):,} 个成员，一个归档只读一遍）…")
        try:
            got = (_pull_zip(archive, want, images_dir, args.dry_run)
                   if zipfile.is_zipfile(archive)
                   else _pull_tar(archive, want, images_dir, args.dry_run))
        except (tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
            print(f"  [失败] {exc}")
            continue
        total += got
        print(f"  {'可解出' if args.dry_run else '已解出'} {got:,} / {len(want):,}")
        if got < len(want):
            print(f"  [注意] 有 {len(want) - got:,} 个成员在归档里找不到，"
                  f"多半是 member 路径和归档内的实际路径对不上")

    print(f"\n合计 {'可补' if args.dry_run else '已补'} {total:,} / {n_missing:,} 张")
    if args.dry_run:
        print("--dry-run，未写入任何文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

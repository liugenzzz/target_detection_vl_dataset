#!/usr/bin/env python3
"""把已有的 VLM 缓存改名到当前的提示词指纹下，避免白跑一遍模型。

    python scripts/migrate_vlm_cache.py --discover                  # 先认指纹
    python scripts/migrate_vlm_cache.py --old-fp <指纹> --dry-run    # 再看命中
    python scripts/migrate_vlm_cache.py --old-fp <指纹>              # 最后改名

【为什么需要这个】缓存键是 md5(图片名|bbox|提示词指纹)。指纹一变，全部缓存
就失效 —— 而缓存文件里只存了描述文本，没存图片名，光看文件名反查不出来。

但图片名我们有（images_dir 里就是），bbox 对构建期那次「挑对象」调用固定为
空列表，所以两个键都能重算：按旧指纹算出老文件名，按当前指纹算出新文件名，
改名即可。一次调用都不用重发。

--old-fp 传缓存【写入时】用的那个指纹，不是现在的。不知道是多少就先跑
--discover：它把 git 历史里 prompts/ 的每个版本都还原出来，用当时的算法各算
一个候选指纹，再拿这批候选去缓存目录里试，命中的那个就是。
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config                       # noqa: E402
from core.vlm_client import (                        # noqa: E402
    MODEL_FACING_PROMPT_DIRS, _prompt_fingerprint,
)
from core.yolo import IMAGE_EXTS                     # noqa: E402

# 认指纹时抽多少张图去试。够区分真假，又不至于把目录翻一遍。
PROBE_SAMPLE = 300


def key_of(image_name: str, fingerprint: Optional[str]) -> str:
    """和 VlmClient._key 保持一致。改那边就要改这里。

    fingerprint 为 None 表示更早的那版键（键里根本没有指纹这一项）。
    """
    raw = f"{image_name}|[]" if fingerprint is None else f"{image_name}|[]|{fingerprint}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------- 指纹候选

def _git(repo: Path, *args: str) -> str:
    return subprocess.run(("git", "-C", str(repo), *args),
                          capture_output=True, text=True, check=True).stdout


def _fp_from_blobs(files: List[Tuple[str, bytes]]) -> str:
    """两版算法共用的最后一步：按给定顺序，把文件名和内容喂进 md5。"""
    h = hashlib.md5()
    for name, blob in files:
        h.update(name.encode())
        h.update(blob)
    return h.hexdigest()[:12]


def _candidates_from_git(prompt_dir: Path, limit: int) -> Dict[str, str]:
    """把 git 历史里 prompts/ 的每个版本还原出来，各算两个候选指纹。

    两个：老算法（整棵 prompts/ 树）和新算法（只算发给模型的那几个子目录）。
    缓存是哪一版写的不知道，所以两边都备着。
    """
    out: Dict[str, str] = {}
    try:
        repo = Path(_git(prompt_dir, "rev-parse", "--show-toplevel").strip())
        rel = prompt_dir.resolve().relative_to(repo.resolve()).as_posix()
        commits = _git(repo, "log", f"-{limit}", "--format=%H %h %s",
                       "--", rel).splitlines()
    except (subprocess.CalledProcessError, ValueError, OSError) as exc:
        print(f"[跳过 git 历史] {exc}")
        return out

    for line in commits:
        sha, short, subject = (line.split(" ", 2) + ["", ""])[:3]
        try:
            listing = _git(repo, "ls-tree", "-r", "--name-only", sha, "--", rel)
        except subprocess.CalledProcessError:
            continue
        paths = sorted(p for p in listing.splitlines() if p.endswith(".txt"))
        if not paths:
            continue
        blobs = {p: _git_blob(repo, sha, p) for p in paths}

        full = [(Path(p).name, blobs[p]) for p in paths]
        out.setdefault(_fp_from_blobs(full), f"{short} 整树 · {subject[:36]}")

        narrow: List[Tuple[str, bytes]] = []
        for sub in MODEL_FACING_PROMPT_DIRS:
            pre = f"{rel}/{sub}/"
            narrow += [(Path(p).name, blobs[p]) for p in paths if p.startswith(pre)]
        if narrow:
            out.setdefault(_fp_from_blobs(narrow), f"{short} 模型可见 · {subject[:36]}")
    return out


def _git_blob(repo: Path, sha: str, path: str) -> bytes:
    return subprocess.run(("git", "-C", str(repo), "show", f"{sha}:{path}"),
                          capture_output=True, check=True).stdout


def _worktree_variants(prompt_dir: Path) -> Dict[str, str]:
    """从【当前工作区的文件】派生候选，而不是从 git 里读。

    为什么还要这一路：git 历史只覆盖「提示词的每个版本都提交过」的情形。
    实际最常见的是另一种 —— 缓存跑完之后拉了一次代码，多出来一个提示词文件，
    别的一个字没动。那时的指纹 = 现在这棵树【去掉那个新文件】。所以除了整树，
    再把「少一个文件」「少一个子目录」的组合也各算一个候选：候选一百来个，
    试一遍不到一秒，却正好盖住这个情形。
    """
    files = [(p, p.name, p.read_bytes()) for p in sorted(prompt_dir.rglob("*.txt"))]
    out: Dict[str, str] = {}

    def add(keep, why: str) -> None:
        full = [(n, b) for p, n, b in files if keep(p)]
        if full:
            out.setdefault(_fp_from_blobs(full), f"本地文件 · 整树 · {why}")
        narrow = []
        for sub in MODEL_FACING_PROMPT_DIRS:
            d = prompt_dir / sub
            narrow += [(n, b) for p, n, b in files
                       if keep(p) and d in p.parents]
        if narrow:
            out.setdefault(_fp_from_blobs(narrow), f"本地文件 · 模型可见 · {why}")

    add(lambda p: True, "原样")
    for path, _, _ in files:
        add(lambda p, x=path: p != x, f"去掉 {path.relative_to(prompt_dir)}")
    for sub in sorted({p.parent for p, _, _ in files}):
        add(lambda p, d=sub: d not in (p, *p.parents),
            f"去掉整个 {sub.relative_to(prompt_dir)}/")
    return out


def discover(cache_dir: Path, images: List[str], extra: List[str],
             prompt_dir: Path, limit: int) -> Optional[str]:
    """拿候选指纹逐个去缓存目录里试，返回命中的那个。"""
    cands: Dict[str, str] = {_prompt_fingerprint(): "当前工作区 · 模型可见（= 现在写入用的）"}
    for fp, why in _worktree_variants(prompt_dir).items():
        cands.setdefault(fp, why)
    for fp, why in _candidates_from_git(prompt_dir, limit).items():
        cands.setdefault(fp, why)
    for fp in extra:
        cands.setdefault(fp, "命令行给的")

    step = max(1, len(images) // PROBE_SAMPLE)
    probe = images[::step][:PROBE_SAMPLE]
    print(f"候选指纹 {len(cands)} 个，各拿 {len(probe)} 张图试：\n")

    scored = []
    for fp, why in cands.items():
        hit = sum(1 for n in probe if (cache_dir / f"{key_of(n, fp)}.json").exists())
        scored.append((hit, fp, why))
    # 老得不能再老的那版键：压根没有指纹
    nofp = sum(1 for n in probe if (cache_dir / f"{key_of(n, None)}.json").exists())
    scored.sort(key=lambda t: -t[0])

    shown = [t for t in scored if t[0]] or scored[:8]
    for hit, fp, why in shown:
        mark = "  <== 就是它" if hit else ""
        print(f"  {fp}  命中 {hit:>4}/{len(probe)}   {why}{mark}")
    if len(shown) < len(scored):
        print(f"  ……另外 {len(scored) - len(shown)} 个候选命中数为 0，略")
    if nofp:
        print(f"  (无指纹)      命中 {nofp:>4}/{len(probe)}   更早的那版缓存键")

    best_hit, best_fp, _ = scored[0]
    if best_hit == 0:
        print("\n[没找到] 所有候选都是 0，说明缓存不是这批提示词写的，或者")
        print("         --images-dir / --cache-dir 这两个目录对不上（见下面的核对）。")
        return None
    print(f"\n认出来了：--old-fp {best_fp}")
    return best_fp


def _sanity(cache_dir: Path, images_dir: Path, images: List[str], total: int) -> None:
    """目录本身对不对，比指纹更值得先看一眼。"""
    print("\n核对一下这两个目录：")
    print(f"  缓存 {cache_dir}   {total:,} 个 json")
    print(f"  图片 {images_dir}   {len(images):,} 张")
    if total and abs(total - len(images)) > max(len(images) * 0.2, 1000):
        print(f"  [注意] 两边数量差了 {abs(total - len(images)):,}，"
              f"很可能不是同一批 —— 先确认缓存目录是不是跑那批数据时用的那个")
    sibs = [d for d in sorted(cache_dir.parent.parent.glob("*/vlm_cache"))
            if d.resolve() != cache_dir.resolve() and d.is_dir()]
    for d in sibs:
        print(f"  [还有别的缓存目录] {d}   {len(list(d.glob('*.json'))):,} 个 json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--old-fp", help="缓存写入时用的提示词指纹；不知道就用 --discover")
    ap.add_argument("--discover", action="store_true",
                    help="从 git 历史里把候选指纹都算出来，试出缓存用的是哪个")
    ap.add_argument("--history", type=int, default=40, help="--discover 往回翻多少个提交")
    ap.add_argument("--cache-dir", help="默认取 config 的 vlm.cache_dir")
    ap.add_argument("--images-dir", help="默认取 config 的 paths.images_dir")
    ap.add_argument("--dry-run", action="store_true", help="只报数不改名")
    args = ap.parse_args()

    if not args.old_fp and not args.discover:
        print("[失败] --old-fp 和 --discover 至少给一个")
        return 1

    cfg = load_config(args.config)
    cache_dir = Path(args.cache_dir or cfg.get_path("vlm.cache_dir", "./output/vlm_cache"))
    images_dir = Path(args.images_dir or cfg.require("paths.images_dir"))
    new_fp = _prompt_fingerprint()

    if not cache_dir.is_dir():
        print(f"[失败] 缓存目录不存在：{cache_dir}")
        return 1
    if not images_dir.is_dir():
        print(f"[失败] 图片目录不存在：{images_dir}")
        return 1

    images = [p.name for p in sorted(images_dir.iterdir())
              if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        print(f"[失败] {images_dir} 里没有图片")
        return 1
    total = len(list(cache_dir.glob("*.json")))

    old_fp = args.old_fp
    if args.discover:
        import prompts as _p
        _sanity(cache_dir, images_dir, images, total)
        print()
        found = discover(cache_dir, images, [args.old_fp] if args.old_fp else [],
                         _p.PROMPT_DIR, args.history)
        if not found:
            return 1
        old_fp = found

    if old_fp == new_fp:
        print(f"\n旧指纹和当前指纹相同（{new_fp}），缓存本来就能命中，无需迁移")
        return 0

    print(f"\n缓存目录 {cache_dir}")
    print(f"图片 {len(images):,} 张   {old_fp} -> {new_fp}\n")

    hit = renamed = already = collision = 0
    for name in images:
        src = cache_dir / f"{key_of(name, old_fp)}.json"
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

    print(f"按旧指纹命中 {hit:,} / {len(images):,} 张图"
          f"（占缓存目录 {total:,} 个文件的 {hit / max(total, 1) * 100:.0f}%）")
    print(f"{'待改名' if args.dry_run else '已改名'} {renamed:,}   目标已存在 {already:,}   失败 {collision:,}")
    if hit == 0:
        print("\n[注意] 一个都没命中，--old-fp 多半传错了。跑 --discover 让脚本自己认。")
        _sanity(cache_dir, images_dir, images, total)
        return 1
    if args.dry_run:
        print("\n--dry-run，未改动任何文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

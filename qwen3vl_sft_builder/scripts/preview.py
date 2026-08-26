#!/usr/bin/env python
"""从构建好的 JSONL 里分层抽样，把框画到图上，供人工复核。

不要对着坐标数字核对 —— 看图。这是复核标注质量唯一靠谱的方式。

分层抽样：按难度(easy/medium/hard)和样本类型(single/multi/negative)各抽一定比例，
避免只看到清晰好看的样本，看不出困难样本上哪里会出问题。

    python scripts/preview.py --jsonl output/train.jsonl -n 60
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config  # noqa: E402

COLORS = {"easy": (60, 200, 120), "medium": (250, 180, 40),
          "hard": (240, 70, 70), None: (120, 160, 255)}


def stratified(rows, n, seed=0):
    """按 (样本类型, 难度) 分层抽样，各层等比例。"""
    buckets = defaultdict(list)
    for r in rows:
        m = r.get("metadata", {})
        buckets[(m.get("sample_type"), m.get("difficulty"))].append(r)
    rng = random.Random(seed)
    per = max(1, n // max(len(buckets), 1))
    out = []
    for key in sorted(buckets, key=lambda k: str(k)):
        group = buckets[key]
        rng.shuffle(group)
        out.extend(group[:per])
    rng.shuffle(out)
    return out[:n]


def draw(row, images_dir: Path, out_dir: Path, scale: int) -> Path | None:
    from PIL import Image, ImageDraw

    img_path = images_dir / Path(row["images"][0]).name
    if not img_path.exists():
        return None
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)

    meta = row.get("metadata", {})
    color = COLORS.get(meta.get("difficulty"), COLORS[None])

    # 从第 4 轮（gpt 的定位回答）里取 bbox
    boxes = []
    for turn in row["conversations"]:
        if turn["from"] != "gpt":
            continue
        try:
            payload = json.loads(turn["value"])
        except (json.JSONDecodeError, TypeError):
            continue
        items = payload if isinstance(payload, list) else [payload]
        for it in items:
            if isinstance(it, dict) and "bbox_2d" in it:
                boxes.append((it["bbox_2d"], it.get("label", "")))

    for bbox, label in boxes:
        x1, y1, x2, y2 = [c / scale for c in bbox]
        px = (x1 * W, y1 * H, x2 * W, y2 * H)
        d.rectangle(px, outline=color, width=max(2, int(min(W, H) * 0.004)))
        d.text((px[0] + 3, max(0, px[1] - 14)), label, fill=color)

    tag = f"{meta.get('sample_type','?')}/{meta.get('difficulty','-')}"
    d.rectangle((0, 0, 260, 18), fill=(0, 0, 0))
    d.text((4, 4), f"{tag}  n={len(boxes)}", fill=(255, 255, 255))

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{row['id']}.jpg"
    im.save(out, quality=88)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--config")
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = load_config(args.config)
    images_dir = Path(cfg.require("paths.images_dir"))
    scale = int(cfg.get_path("coords.scale", 1000))
    out_dir = Path(args.out or Path(cfg.get_path("paths.output_dir", "./output")) / "verify")

    rows = [json.loads(l) for l in Path(args.jsonl).open(encoding="utf-8")]
    picked = stratified(rows, args.n, int(cfg.get_path("sampling.seed", 0)))

    manifest = []
    drawn = 0
    for row in picked:
        p = draw(row, images_dir, out_dir, scale)
        if p:
            drawn += 1
        manifest.append({
            "id": row["id"], "image": row["images"][0],
            "type": row.get("metadata", {}).get("sample_type"),
            "difficulty": row.get("metadata", {}).get("difficulty"),
            "verify_image": str(p) if p else None,
            "conversations": row["conversations"],
        })
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"抽样 {len(picked)} 条，画出 {drawn} 张验证图 -> {out_dir}")
    print(f"对话内容见 {out_dir / 'manifest.json'}")
    print("\n复核要点：框有没有套准物体 / 指代能不能唯一指到那个目标 / 描述有没有编造")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

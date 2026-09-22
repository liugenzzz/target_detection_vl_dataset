#!/usr/bin/env python3
"""把清单里人写的【指代短语】直接做成样本，不经 YOLO 标注、不调 VLM。

    python scripts/build_expressions.py --dry-run     # 只统计能出多少
    python scripts/build_expressions.py

【这批数据长什么样】有一类源（sky）的标注不是「类别 + 框」，而是
「框 + 一句人写的指代短语」：

    {"label": null, "bbox_xyxy": [625, 635, 799, 1080],
     "expressions": ["A man wearing a brown leather jacket and beige pants
                      stands near the center of the scene, giving a thumbs-up
                      gesture. He is in front of a gray box, with a woman in
                      a red dress to his right."]}

label 是 null —— 所以它生不出 YOLO 标注（第一个字段就是 class_id），
上游的转换脚本对这批只能写出空文件，22,473 张图整个作废。

【为什么单独走一条路，不塞进主流程】主流程的一切都挂在类别名上：
_same_label 判同类密集度、count_class 数个数、detect_class 穷举、
exist_negative 挑易混类别。给这些框编一个假类别名，上面每一项都会得出
错误答案，而且看不出错。不如不进那条线。

【为什么值】这些短语是人写的，而主线的指代短语现在是模型现场生成的 ——
人写的质量更高，且【零调用成本】。更巧的是它们天然带外观 + 方位 + 周边关系
（看上面那个例子），正好是主线末轮要的那种描述。所以一个框上有两条以上
短语时，可以一条当指代、另一条当描述，凑成完整的三段式主线，仍然零成本。

产出两个任务：
    refer_ground     该框有 >= 2 条短语：短语A -> 框 -> 短语B（三段式，主线）
    region_describe  只有 1 条短语：框 -> 短语（单轮）

输出目录单独配，跑完用 scripts/merge_by_group.py 合并进主数据集。
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prompts                                          # noqa: E402
from config import load_config                          # noqa: E402
from core import progress                               # noqa: E402
from core.coords import yolo_to_bbox2d                  # noqa: E402
from core.difficulty import REJECT, Grader              # noqa: E402
from core.pipeline import (                             # noqa: E402
    _focus_difficulty, _image_value, _split_by_source, _write_jsonl,
)
from core.sample import IMAGE_TOKEN, validate_sample    # noqa: E402
from core.yolo import Box, index_images                 # noqa: E402

# 这条路产出的两个任务名。主线的定义是「指代 -> 坐标 -> 描述」三段齐全，
# 所以只有 refer_ground 算主线；region_describe 只有后半截。
MAIN_LINE = ("refer_ground",)


def _rows(manifest: Path):
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _exprs(ann: dict) -> List[str]:
    """一个框上的指代短语，去重去空，保持原顺序。

    末尾的英文句号去掉：短语要被「」括着嵌进中文问句，留着会变成 "…。」。"
    这种双标点。句子的收尾标点由模板给，不由数据给。
    """
    out: List[str] = []
    for e in (ann.get("expressions") or []):
        e = str(e).strip().rstrip(".。 ")
        if e and e not in out:
            out.append(e)
    return out


def _boxes(row: dict):
    """把 bbox_xyxy（像素）转成 Box（0~1 归一化）。返回 (boxes, 每框的短语)。"""
    q = row.get("quality") or {}
    w, h = int(q.get("width") or 0), int(q.get("height") or 0)
    if w <= 0 or h <= 0:
        return [], []
    boxes, exprs = [], []
    for ann in (row.get("annotations") or []):
        es = _exprs(ann)
        bb = ann.get("bbox_xyxy")
        if not es or not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bb)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(Box(len(boxes), -1, "", (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                         (x2 - x1) / w, (y2 - y1) / h))
        exprs.append(es)
    return boxes, exprs


def _turns(*pairs):
    out = []
    for i, (q, a) in enumerate(pairs):
        out.append({"from": "human", "value": (f"{IMAGE_TOKEN}\n{q}" if i == 0 else q)})
        out.append({"from": "gpt", "value": a})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--manifest", type=Path, help="默认取 paths.selection_manifest")
    ap.add_argument("--out-dir", type=Path, help="默认 <output_dir>_expr")
    ap.add_argument("--limit", type=int, help="只处理前 N 张图（试跑用）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    args = ap.parse_args()

    cfg = load_config(args.config)
    manifest = Path(args.manifest or cfg.require("paths.selection_manifest"))
    images_dir = Path(cfg.require("paths.images_dir"))
    out_dir = Path(args.out_dir or (str(cfg.get_path("paths.output_dir", "./output")) + "_expr"))
    grader = Grader(cfg)
    for conflict in grader.config_conflicts():
        print(f"[配置冲突] {conflict}")
    scale = int(cfg.get_path("coords.scale", 1000))
    origin = int(cfg.get_path("coords.origin", 0))
    fence = bool(cfg.get_path("output.wrap_json_in_code_block", False))
    cap = int(cfg.get_path("sampling.samples_per_image_cap", 8))
    rng = random.Random(int(cfg.get_path("sampling.seed", 20260826)))
    forbid = tuple(cfg.get_path("phrase_banks.forbid_global", []) or [])
    style = str(cfg.get_path("output.image_path_style", "filename"))

    index = index_images(images_dir)
    samples: List[Dict[str, Any]] = []
    made = collections.Counter()
    skip = collections.Counter()
    n_img = n_box = 0

    rows = list(_rows(manifest))
    bar = progress.make("扫描清单", len(rows), True)
    for row in rows:
        bar.step()
        if args.limit and n_img >= args.limit:
            break
        stem = Path(str(row.get("selected_image") or "")).stem
        image_path = index.get(stem)
        if not stem or image_path is None:
            skip["找不到图片"] += 1
            continue
        boxes, exprs = _boxes(row)
        if not boxes:
            skip["没有带短语的框"] += 1
            continue
        n_img += 1
        n_box += len(boxes)

        q = row.get("quality") or {}
        img_w, img_h = int(q["width"]), int(q["height"])
        gmap = {g.box_index: g for g in grader.grade_image(boxes, img_w, img_h)}
        usable = [b for b in boxes if gmap[b.index].grade != REJECT]
        if not usable:
            skip["框全被质量过滤"] += 1
            continue

        rng.shuffle(usable)
        for b in usable[:cap]:
            es = exprs[b.index]
            bbox = yolo_to_bbox2d(b.cx, b.cy, b.w, b.h, img_w, img_h, scale, origin)
            body = json.dumps({"bbox_2d": bbox}, ensure_ascii=False)
            answer = f"```json\n{body}\n```" if fence else body
            if len(es) >= 2:
                # 一条当指代、另一条当描述 —— 三段式主线，零调用成本
                refer, desc = es[0], es[1]
                task = "refer_ground"
                convs = _turns(
                    (prompts.render_choice("expr_ground", rng, expr=refer), answer),
                    (prompts.render_choice("expr_describe", rng,
                                           bbox=json.dumps(bbox, ensure_ascii=False)), desc))
            else:
                task = "region_describe"
                convs = _turns((prompts.render_choice(
                    "expr_describe", rng,
                    bbox=json.dumps(bbox, ensure_ascii=False)), es[0]))

            sample = {
                "id": f"{stem}_{task}_{made[task]}",
                "images": [_image_value(image_path, style)],
                "conversations": convs,
                "metadata": {
                    "task_type": task,
                    "is_main_line": task in MAIN_LINE,
                    "answer_format": "normal",
                    "source_image": image_path.name,
                    "source_annotation": "",
                    "image_width": img_w, "image_height": img_h,
                    "coordinate_mode": f"qwen_relative_{scale}", "bbox_scale": scale,
                    "label": None,          # 这条路本来就没有类别
                    "focus_box_indices": [b.index],
                    **_focus_difficulty([b.index], gmap, grader),
                    "n_turns": len(convs) // 2,
                    "expression_source": "human",
                    "n_expressions": len(es),
                },
            }
            issues = validate_sample(sample, forbid)
            if issues:
                skip["落盘校验不过"] += 1
                continue
            made[task] += 1
            samples.append(sample)
    bar.close()

    print(f"\n清单 {len(rows):,} 行   可用图片 {n_img:,}   带短语的框 {n_box:,}")
    if skip:
        print("跳过：")
        for k, v in skip.most_common():
            print(f"    {k:16} {v:,}")
    if not samples:
        print("\n一条都没出。检查 manifest 里是不是真的有 expressions 字段。")
        return 1

    total = len(samples)
    main = sum(made[t] for t in MAIN_LINE)
    print(f"\n产出 {total:,} 条（每图 {total / max(n_img, 1):.2f} 条）")
    for t, n in made.most_common():
        mark = " ←主线（三段式）" if t in MAIN_LINE else ""
        print(f"    {t:18} {n:>8,}  {n / total * 100:5.1f}%{mark}")
    print(f"  其中主线 {main / total * 100:.1f}%")

    if args.dry_run:
        print("\n--dry-run，未落盘")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = _split_by_source(samples, cfg, int(cfg.get_path("sampling.seed", 20260826)))
    meta = bool(cfg.get_path("output.include_metadata", True))
    _write_jsonl(out_dir / "train.jsonl", train, meta)
    _write_jsonl(out_dir / "val.jsonl", val, meta)
    _write_jsonl(out_dir / "test.jsonl", test, True)
    print(f"\n写入 {out_dir}   train {len(train):,} / val {len(val):,} / test {len(test):,}")
    print(f"合并进主数据集：python scripts/merge_by_group.py --into <主 output> --add {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

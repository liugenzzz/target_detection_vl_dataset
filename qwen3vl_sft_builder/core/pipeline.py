"""构建管道：标注目录 -> 质量过滤 -> VLM 挑对象 -> 按配比生成十种任务 -> JSONL。"""

from __future__ import annotations

import json
import logging
import random
import collections
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import prompts

from .builder import validate_sample
from .classes import load_class_table
from .coords import yolo_to_bbox2d
from .difficulty import REJECT, Grader
from .grouping import source_group_key
from .referring import spatial_phrase
from .tasks import MAIN_LINE, TASKS, Ctx


def _load_measure_words(cfg) -> Dict[str, str]:
    """读类别量词表。缺失时返回空 dict，构建时全部退回「个」。"""
    import yaml
    rel = str(cfg.get_path("measure_words_path", "") or "")
    if not rel:
        return {}
    path = Path(rel)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.exists():
        logger.info("未找到量词表 %s，量词一律用「个」。"
                    "跑 scripts/build_measure_words.py 可生成。", path)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in (data.get("measure_words") or {}).items()}
from .vlm_client import VlmClient
from .yolo import iter_annotations

logger = logging.getLogger(__name__)

# 一个样本槽位最多尝试几个任务类型。太小会浪费槽位，太大会让配比偏离权重。
MAX_TRY = 4


def _weighted_order(weights: Dict[str, float], rng: random.Random) -> List[str]:
    """按权重把任务名铺成一个抽样池。权重为 0 的任务不参与。"""
    pool: List[str] = []
    for name, w in weights.items():
        if name in TASKS and w > 0:
            pool.extend([name] * int(round(float(w) * 10)))
    rng.shuffle(pool)
    return pool or list(TASKS)


def _box_list_text(boxes, bbox2d) -> str:
    lines = [f"图中共 {len(boxes)} 个已标注目标："]
    for b in boxes:
        lines.append(f"  [{b.index}] {b.label}  位于 {bbox2d(b)}")
    return "\n".join(lines)


def build(cfg, limit: int | None = None) -> Dict[str, Any]:
    labels_dir = Path(cfg.require("paths.labels_dir"))
    images_dir = Path(cfg.require("paths.images_dir"))
    output_dir = Path(cfg.get_path("paths.output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    table = load_class_table(cfg.require("paths.classes_yaml"))
    grader = Grader(cfg)
    vlm = VlmClient(cfg)

    seed = int(cfg.get_path("sampling.seed", 20260826))
    cap = int(cfg.get_path("sampling.samples_per_image_cap", 8))
    sanity = int(cfg.get_path("quality.sanity_max_boxes", 1000))
    scale = int(cfg.get_path("coords.scale", 1000))
    origin = int(cfg.get_path("coords.origin", 0))
    short_ratio = float(cfg.get_path("short_answer_ratio", 0.14))
    max_pick = int(cfg.get_path("sampling.vlm_max_pick", 6))
    weights = cfg.get_path("tasks", {}) or {}
    measure_words = _load_measure_words(cfg)
    require_desc = bool(cfg.get_path("main_line_requires_description", True))
    min_desc_len = int(cfg.get_path("min_description_len", 18))
    all_labels = sorted(table.id2name.values())

    rng = random.Random(seed)
    task_pool = _weighted_order(weights, rng)

    # ---- 阶段一：扫描 + 质量过滤 ----
    scenes: List[Dict[str, Any]] = []
    n_images = n_boxes = 0
    for ann in iter_annotations(labels_dir, images_dir, table, sanity):
        n_images += 1
        n_boxes += len(ann.boxes)
        graded = grader.grade_image(ann.boxes, ann.width, ann.height)
        gmap = {g.box_index: g for g in graded}
        kept = [b for b in ann.boxes if gmap[b.index].grade != REJECT]
        if not kept:
            continue
        scenes.append({"ann": ann, "kept": kept, "gmap": gmap})
        if limit and len(scenes) >= limit:
            break

    if not scenes:
        raise RuntimeError(f"{labels_dir} 下没有可用场景，检查类别表与阈值配置")

    def bbox2d_for(ann):
        return lambda b: yolo_to_bbox2d(b.cx, b.cy, b.w, b.h, ann.width, ann.height,
                                        scale, origin)

    # ---- 阶段二：并发预取 VLM（一张图一次调用，挑对象+属性+描述）----
    if vlm.enabled:
        tasks = []
        for sc in scenes:
            ann, kept = sc["ann"], sc["kept"]
            text = prompts.render("vlm_select",
                                  box_list=_box_list_text(kept, bbox2d_for(ann)),
                                  max_pick=min(max_pick, len(kept)))
            tasks.append((ann.image_path, [ann.width, ann.height], "scene", text))
        vlm.prefetch(tasks)

    # ---- 阶段三：按配比生成 ----
    samples: List[Dict[str, Any]] = []
    made = Counter()
    failed = Counter()
    invalid = 0
    cursor = 0

    for sc in scenes:
        ann, kept, gmap = sc["ann"], sc["kept"], sc["gmap"]
        b2d = bbox2d_for(ann)
        vlm_info = vlm.scene_info(ann.image_path, [ann.width, ann.height],
                                 {b.index for b in kept})
        used: set = set()          # 本图已出过样本的框，避免同一目标反复出样本
        # 指代骨架按图算一次：全图唯一性校验需要看到该图全部目标
        label_counts = collections.Counter(b.label for b in kept)
        n_want = min(cap, max(1, len(kept)))
        for _ in range(n_want):
            # 一个槽位最多试 MAX_TRY 个任务：某个任务在这张图上条件不满足时
            # （比如没有单实例类别、VLM 没挑中可描述的目标），顺延到池里下一个，
            # 而不是白白浪费这个槽位。失败次数仍单独统计。
            out = None
            name = ""
            for _try in range(MAX_TRY):
                name = task_pool[cursor % len(task_pool)]
                cursor += 1
                ctx = Ctx(annotation=ann, boxes=kept, grades=gmap, vlm=vlm_info,
                          all_labels=all_labels,
                          bbox2d=b2d, spatial=lambda b: spatial_phrase(b.cx, b.cy),
                          rng=rng, short_answer=rng.random() < short_ratio,
                          measure_words=measure_words, require_desc=require_desc,
                          used=used, min_desc_len=min_desc_len)
                try:
                    out = TASKS[name](ctx)
                except Exception as exc:                   # noqa: BLE001
                    logger.warning("任务 %s 生成异常：%s", name, exc)
                    out = None
                if out is not None:
                    break
                failed[name] += 1
            if out is None:
                continue

            sample = {
                "id": f"{ann.stem}_{name}_{made[name]}",
                "images": [ann.image_path.name],
                "conversations": out["conversations"],
                "metadata": {
                    "task_type": name,
                    "is_main_line": name in MAIN_LINE,
                    # 按问句是否真的带了短答案要求来标记 —— 定位类任务的答案是
                    # bbox JSON，不会追加「用一个词回答」，那时即使抽中了短答案
                    # 也应标 normal，否则这个字段筛不出真正的短答案样本。
                    "answer_format": (
                        "short"
                        if prompts.load("short_answer_suffix") in out["conversations"][0]["value"]
                        else "normal"),
                    "source_image": ann.image_path.name,
                    "source_annotation": ann.label_path.name,
                    "image_width": ann.width, "image_height": ann.height,
                    "coordinate_mode": f"qwen_relative_{scale}", "bbox_scale": scale,
                    "label": out.get("label"),
                    "focus_box_indices": out.get("focus", []),
                    "n_turns": len(out["conversations"]) // 2,
                    **{k: v for k, v in out.items()
                       if k in ("attribute", "relation", "count", "polarity",
                                "question_source", "n_boxes",
                                "inventory")},
                },
            }
            issues = validate_sample(sample)
            if issues:
                invalid += 1
                logger.warning("样本 %s 校验失败：%s", sample["id"], issues)
                continue
            made[name] += 1
            used.update(out.get("focus", []))
            samples.append(sample)

    # ---- 阶段四：按来源分组划分 ----
    train, val = _split_by_source(samples, cfg, seed)
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "val.jsonl", val)

    total = sum(made.values()) or 1
    stats = {
        "images_scanned": n_images,
        "boxes_total": n_boxes,
        "scenes_usable": len(scenes),
        "samples_total": sum(made.values()),
        "by_task_type": {k: made[k] for k in TASKS if made[k]},
        "task_ratio_actual": {k: round(made[k] / total, 4) for k in TASKS if made[k]},
        "main_line_ratio": round(sum(made[k] for k in MAIN_LINE) / total, 4),
        "short_answer_ratio_actual": round(
            sum(1 for s in samples if s["metadata"]["answer_format"] == "short") / total, 4),
        "task_unavailable": {k: v for k, v in failed.items() if v},
        "invalid_dropped": invalid,
        "vlm_calls": dict(vlm.stats),
        "split": _split_stats(train, val),
        "classes_total": table.count,
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["output"] = {"train": str(output_dir / "train.jsonl"),
                       "val": str(output_dir / "val.jsonl")}
    return stats


def _split_by_source(samples, cfg, seed) -> Tuple[List, List]:
    """按【原始来源】分组划分。数据含视频抽帧与 Roboflow 增强，
    按图片随机划分会让同源图泄漏进验证集，指标虚高且难以察觉。"""
    ratio = float(cfg.get_path("split.val_ratio", 0.05))
    groups: Dict[str, List] = {}
    for s in samples:
        groups.setdefault(source_group_key(s["images"][0]), []).append(s)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = len(samples) * ratio
    val: List = []
    train: List = []
    for k in keys:
        (val if len(val) < target else train).extend(groups[k])
    return train, val


def _split_stats(train, val) -> Dict[str, Any]:
    tg = {source_group_key(s["images"][0]) for s in train}
    vg = {source_group_key(s["images"][0]) for s in val}
    return {"train": len(train), "val": len(val),
            "train_groups": len(tg), "val_groups": len(vg),
            "group_overlap": len(tg & vg)}


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

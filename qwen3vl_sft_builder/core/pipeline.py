"""构建管道编排：标注目录 -> 分级 -> 配额 -> 三轮样本 -> 按来源划分 -> JSONL。"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .builder import SampleBuilder, validate_sample
from .classes import load_class_table
from .difficulty import Grader, balance_hard_quota, pick_candidates
from .grouping import source_group_key
from .vlm_client import VlmClient
from .yolo import iter_annotations

logger = logging.getLogger(__name__)


def build(cfg, limit: int | None = None) -> Dict[str, Any]:
    """跑完整构建。limit 用于试跑（只处理前 N 张图）。"""
    labels_dir = Path(cfg.require("paths.labels_dir"))
    images_dir = Path(cfg.require("paths.images_dir"))
    output_dir = Path(cfg.get_path("paths.output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    table = load_class_table(cfg.require("paths.classes_yaml"))
    grader = Grader(cfg)
    vlm = VlmClient(cfg)
    builder = SampleBuilder(cfg, table, vlm)

    seed = int(cfg.get_path("sampling.seed", 20260826))
    cap = int(cfg.get_path("sampling.samples_per_image_cap", 8))
    hard_quota = float(cfg.get_path("difficulty.hard_quota", 0.10))
    multi_ratio = float(cfg.get_path("sampling.multi_target_ratio", 0.10))
    multi_max = int(cfg.get_path("sampling.multi_target_max", 3))
    neg_ratio = float(cfg.get_path("sampling.negative_ratio", 0.05))
    sanity = int(cfg.get_path("quality.sanity_max_boxes", 1000))

    # ---- 阶段一：逐图分级、挑候选 ----
    annotations: Dict[str, Any] = {}
    candidates: List[Tuple[str, Any]] = []
    grade_hist = Counter()
    n_images = n_boxes = 0

    for ann in iter_annotations(labels_dir, images_dir, table, sanity):
        n_images += 1
        n_boxes += len(ann.boxes)
        graded = grader.grade_image(ann.boxes, ann.width, ann.height)
        for g in graded:
            grade_hist[g.grade] += 1
        annotations[ann.stem] = (ann, {g.box_index: g for g in graded})
        for g in pick_candidates(graded, cap, seed):
            candidates.append((ann.stem, g))
        if limit and n_images >= limit:
            break

    if not candidates:
        raise RuntimeError(f"{labels_dir} 下没有产出任何候选目标，检查类别表与阈值配置")

    # ---- 阶段二：全局配额，把困难目标压到 hard_quota ----
    selected = balance_hard_quota(candidates, lambda c: c[1].grade, hard_quota, seed)

    # ---- 阶段三：并发预取 VLM 描述 ----
    # 必须在组装之前做完：组装是串行的，逐条调用十万次要 6 天。
    # 预取把结果灌进缓存，组装时全部命中缓存。已缓存的自动跳过，支持断点续跑。
    rng = random.Random(seed)
    by_image: Dict[str, List] = {}
    for stem, g in selected:
        by_image.setdefault(stem, []).append(g)

    if vlm.enabled:
        tasks = []
        for stem, grades in by_image.items():
            ann, _ = annotations[stem]
            box_map = {b.index: b for b in ann.boxes}
            for g in grades:
                tasks.append(builder.vlm_task(ann, box_map[g.box_index], g))
        vlm.prefetch(tasks)

    # 多目标样本数必须全局算，不能「每图 N% 概率」—— 一张图会出多条单目标样本，
    # 逐图概率会被稀释（实测 VisDrone 上 10% 的逐图概率只得到 2.3% 的多目标占比）。
    # 设目标总数 N、多目标样本 m 条（每条消耗约 k 个目标），则
    #     总样本 = m + (N - k*m)，要 m/总样本 = r  =>  m = r*N / (1 + r*(k-1))
    n_targets = len(selected)
    k_avg = (2 + multi_max) / 2
    n_multi = int(round(multi_ratio * n_targets / (1 + multi_ratio * (k_avg - 1))))
    eligible = [stem for stem, gs in by_image.items() if len(gs) >= 2]
    rng.shuffle(eligible)
    multi_stems = set(eligible[:n_multi])

    # ---- 阶段四：组装样本（全部命中缓存，不再发请求）----
    samples: List[Dict[str, Any]] = []
    invalid = 0
    multi_rejected = 0
    for stem, grades in by_image.items():
        ann, grade_map = annotations[stem]
        box_map = {b.index: b for b in ann.boxes}
        grades = sorted(grades, key=lambda g: g.box_index)

        # 多目标样本：按比例抽，一次覆盖 2~3 个目标
        used: set = set()
        if stem in multi_stems:
            k = min(rng.randint(2, multi_max), len(grades))
            # 两条选取偏好：
            # (a) 优先挑【显眼、好描述】的目标 —— easy 档是大、孤立、分区内唯一的
            #     目标，VLM 一两个特征就能锁定，指代自然就短。多目标样本要把
            #     两三个指代拼成一句问题，用难描述的目标会拼出读不下去的长句。
            # (b) 空间分区互不相同，这样模板指代天然互异；同分区的目标只有在
            #     VLM 给出互异视觉指代时才可用，最终由 build_multi 校验。
            _rank = {"easy": 0, "medium": 1, "hard": 2}
            chosen, seen_zones = [], set()
            for g in sorted(grades, key=lambda x: _rank.get(x.grade, 3)):
                if g.zone in seen_zones:
                    continue
                seen_zones.add(g.zone)
                chosen.append(g)
                if len(chosen) >= k:
                    break
            if len(chosen) < 2:
                # 全图目标都挤在同一分区，凑不出互异指代。必须计数，
                # 否则密集数据上多目标占比会悄悄塌掉而报告里看不出来。
                multi_rejected += 1
            else:
                s = builder.build_multi(
                    ann, [box_map[g.box_index] for g in chosen], chosen)
                if s is None:
                    multi_rejected += 1          # 指代无法互异，退回单目标
                else:
                    issues = validate_sample(s)
                    if issues:
                        invalid += 1
                        logger.warning("样本 %s 校验失败：%s", s.get("id"), issues)
                    else:
                        used = {g.box_index for g in chosen}
                        samples.append(s)

        for g in grades:
            if g.box_index in used:
                continue
            s = builder.build_single(ann, box_map[g.box_index], g)
            issues = validate_sample(s)
            if issues:
                invalid += 1
                logger.warning("样本 %s 校验失败：%s", s.get("id"), issues)
            else:
                samples.append(s)

    # ---- 阶段五：拒答样本 ----
    all_labels = sorted({b.label for ann, _ in annotations.values() for b in ann.boxes})
    n_neg = int(len(samples) * neg_ratio)
    neg_added = 0
    if n_neg and len(all_labels) > 1:
        stems = list(by_image)
        rng.shuffle(stems)
        for stem in stems:
            if neg_added >= n_neg:
                break
            ann, _ = annotations[stem]
            present = {b.label for b in ann.boxes}
            absent = [l for l in all_labels if l not in present]
            if not absent:
                continue
            s = builder.build_negative(ann, rng.choice(absent))
            if not validate_sample(s):
                samples.append(s)
                neg_added += 1

    # ---- 阶段六：按来源分组划分 train/val ----
    train, val = _split_by_source(samples, cfg, seed)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, val)

    stats = _stats(samples, train, val, grade_hist, n_images, n_boxes,
                   len(candidates), invalid, neg_added, vlm.stats, table)
    stats["multi_rejected_ambiguous_referring"] = multi_rejected
    stats["vlm_referring_leaked_label"] = builder.leaked_referrings
    stats["vlm_referring_too_long"] = builder.overlong_referrings
    stats["multi_actual_ratio"] = round(
        sum(1 for s in samples if s.get("metadata", {}).get("sample_type") == "multi")
        / max(len(samples), 1), 4)
    stats["output"] = {"train": str(train_path), "val": str(val_path)}
    (output_dir / "build_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def _split_by_source(samples, cfg, seed) -> Tuple[List, List]:
    """按【原始来源】分组划分。数据含视频抽帧与 Roboflow 增强，
    按图片随机划分会让同源图泄漏进验证集，指标虚高且难以察觉。"""
    val_ratio = float(cfg.get_path("split.val_ratio", 0.05))
    if not cfg.get_path("split.group_by_source", True):
        rng = random.Random(seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * val_ratio)
        return shuffled[cut:], shuffled[:cut]

    groups: Dict[str, List] = {}
    for s in samples:
        key = source_group_key(s["metadata"]["source_image"]
                               if "metadata" in s else s["images"][0])
        groups.setdefault(key, []).append(s)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = len(samples) * val_ratio
    val: List = []
    train: List = []
    for k in keys:
        (val if len(val) < target else train).extend(groups[k])
    return train, val


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stats(samples, train, val, grade_hist, n_images, n_boxes,
           n_candidates, invalid, neg_added, vlm_stats, table) -> Dict[str, Any]:
    kinds = Counter(s["metadata"]["sample_type"] for s in samples if "metadata" in s)
    diffs = Counter(s["metadata"].get("difficulty") for s in samples
                    if "metadata" in s and s["metadata"].get("sample_type") == "single")
    total_diff = sum(diffs.values()) or 1
    confusable = sum(1 for s in samples
                     if s.get("metadata", {}).get("confusable_class"))
    train_g = {source_group_key(s["images"][0]) for s in train}
    val_g = {source_group_key(s["images"][0]) for s in val}

    return {
        "images_scanned": n_images,
        "boxes_total": n_boxes,
        "grade_distribution_all_boxes": dict(grade_hist),
        "candidates_after_per_image_cap": n_candidates,
        "samples_total": len(samples),
        "samples_by_type": dict(kinds),
        "single_sample_difficulty": dict(diffs),
        "hard_ratio_in_single": round(diffs.get("hard", 0) / total_diff, 4),
        "negative_samples": neg_added,
        "confusable_class_samples": confusable,
        "invalid_dropped": invalid,
        "vlm_calls": dict(vlm_stats),
        "split": {
            "train": len(train), "val": len(val),
            "train_groups": len(train_g), "val_groups": len(val_g),
            "group_overlap": len(train_g & val_g),
        },
        "classes_total": table.count,
        "confusable_groups": table.confusable_summary(),
    }

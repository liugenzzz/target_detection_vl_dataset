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

from .sample import validate_sample
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
from . import colorcheck, consistency, phrase_bank
from .vlm_client import VlmClient
from .yolo import iter_annotations

logger = logging.getLogger(__name__)

# 一个样本槽位最多尝试几个任务类型。太小会浪费槽位，太大会让配比偏离权重。
MAX_TRY = 4



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
    bank_stats = phrase_bank.install(cfg)
    forbid_chat = tuple(cfg.get_path("phrase_banks.forbid_global", []) or [])
    require_desc = bool(cfg.get_path("main_line_requires_description", True))
    min_desc_len = int(cfg.get_path("min_description_len", 18))
    all_labels = sorted(table.id2name.values())
    color_gate = bool(cfg.get_path("quality.verify_color_with_pixels", True))
    color_stats = Counter()
    color_dropped: List[Dict[str, Any]] = []      # 抽样留证，进构建报告
    image_style = str(cfg.get_path("output.image_path_style", "filename"))
    include_meta = bool(cfg.get_path("output.include_metadata", True))
    json_fence = bool(cfg.get_path("output.wrap_json_in_code_block", False))
    # 名称维度的易混表，供 exist_negative 挑 hard negative。
    # 【剔除上下位词】：图里有遮阳三轮车，问「有没有三轮车」答「没有」是错的。
    # 剩下的是一字之差的并列类别（切管器 vs 切管机），答「没有」才成立。
    confusable, hypernym = {}, {}
    for cid, name in table.id2name.items():
        hyper = set(table.hypernym_group(cid))
        group = [g for g in table.confusable_group(cid)
                 if g != name and g not in hyper]
        if group:
            confusable[name] = group
        if hyper:
            hypernym[name] = sorted(hyper - {name})

    rng = random.Random(seed)
    total_w = sum(v for v in weights.values() if v) or 1
    target = {k: v / total_w for k, v in weights.items() if v}
    if not target:
        raise RuntimeError("config 的 tasks 段里所有权重都是 0，没有任务可生成")
    strict_ratio = str(cfg.get_path("tasks_ratio_mode", "strict")) == "strict"

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
        raw_counts = Counter(b.label for b in ann.boxes)
        scenes.append({"ann": ann, "kept": kept, "gmap": gmap,
                       "raw_counts": dict(raw_counts)})
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
            # 每张图换一种描述的起手方式。固定的例子会把模型的句式钉在那几条上，
            # 十万条描述全是一个套路。轮换之后各种起手方式在整批数据里均匀铺开。
            # 效果用 scripts/dataset_stats.py 的「答案开头集中度」验。
            rule, example = prompts.pick_pair("desc_opening", rng)
            text = prompts.render("vlm_select",
                                  box_list=_box_list_text(kept, bbox2d_for(ann)),
                                  max_pick=min(max_pick, len(kept)),
                                  opening_rule=rule, opening_example=example)
            tasks.append((ann.image_path, [ann.width, ann.height], "scene", text))
        vlm.prefetch(tasks)

    # ---- 阶段三：按配比生成 ----
    samples: List[Dict[str, Any]] = []
    made = Counter()
    failed = Counter()
    invalid = 0

    for sc in scenes:
        ann, kept, gmap = sc["ann"], sc["kept"], sc["gmap"]
        b2d = bbox2d_for(ann)
        vlm_info = vlm.scene_info(ann.image_path, [ann.width, ann.height],
                                 {b.index for b in kept})
        # 【像素核对颜色】模型说「白色车身的卡车」而框落在蓝色卡车上时，
        # 指代就指向了错误的目标，attribute_qa 更是直接答错。颜色能从像素量出来，
        # 不该靠模型自觉。对不上就把这个颜色说法丢掉（其余字段照用），
        # 判据刻意宽松，只拦明显冲突。
        if color_gate:
            color_stats["checked"] += _drop_bad_colors(
                ann, kept, vlm_info, color_stats, color_dropped)
        used: set = set()          # 本图已出过样本的框，避免同一目标反复出样本
        # 指代骨架按图算一次：全图唯一性校验需要看到该图全部目标
        label_counts = collections.Counter(b.label for b in kept)
        n_want = min(cap, max(1, len(kept)))
        for _ in range(n_want):
            # 【按产出缺口调度，不是按槽位轮转】。
            #
            # 各任务的可用率差得很远：主线要求类别在原始标注里唯一、要求 VLM
            # 挑中了目标，约一半槽位填不上；exist_negative 几乎永远能填。
            # 按权重轮转发槽位的话，填不上的那些槽位会被永远可用的任务接走 ——
            # 实测配比给主线 70%，实得 41%，exist_negative 配 7% 拿到 31.9%。
            #
            # 改成每个槽位挑【当前欠账最多】的任务：欠得越多越优先，一旦补上
            # 就轮到别人。这是个自校正的反馈，任何任务的可用率再低也不会被
            # 别人挤掉份额，也不会有任务超发。缺口相同时随机打散，避免固定顺序。
            done_all = sum(made.values())
            deficit = sorted(target,
                             key=lambda t: (made[t] - target[t] * (done_all + 1),
                                            rng.random()))

            def make_ctx():
                return Ctx(annotation=ann, boxes=kept, grades=gmap, vlm=vlm_info,
                           all_labels=all_labels,
                           bbox2d=b2d, spatial=lambda b: spatial_phrase(b.cx, b.cy),
                           rng=rng, short_answer=rng.random() < short_ratio,
                           measure_words=measure_words, require_desc=require_desc,
                           used=used, min_desc_len=min_desc_len,
                           raw_counts=sc["raw_counts"], forbid_chat=forbid_chat,
                           confusable=confusable, hypernym=hypernym,
                          json_fence=json_fence)

            # strict：只试欠账最多的那一个，补不上就空过这个槽位，配比一分不歪。
            # fill：往下顺延，优先填满槽位，配比会偏。数据量的瓶颈从来不在
            # 单张图能出几条，而在有多少张图，所以默认 strict。
            out = None
            name = ""
            for name in deficit[:1 if strict_ratio else MAX_TRY]:
                try:
                    out = TASKS[name](make_ctx())
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
                "images": [_image_value(ann.image_path, image_style)],
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
                       if k in ("attribute", "attribute_kind", "relation",
                                "relation_axis", "count", "polarity",
                                "hard_negative", "question_source", "n_boxes",
                                "inventory")},
                },
            }
            issues = validate_sample(sample, forbid_chat)
            if issues:
                invalid += 1
                logger.warning("样本 %s 校验失败：%s", sample["id"], issues)
                continue
            made[name] += 1
            used.update(out.get("focus", []))
            samples.append(sample)

    # ---- 阶段四：按来源分组划分 ----
    train, val = _split_by_source(samples, cfg, seed)
    _write_jsonl(output_dir / "train.jsonl", train, include_meta)
    _write_jsonl(output_dir / "val.jsonl", val, include_meta)

    total = sum(made.values()) or 1
    # 跨任务一致性核对：同一张图，八个任务说出来的目标数量必须对得上
    kept_labels = {str(sc["ann"].image_path): dict(Counter(b.label for b in sc["kept"]))
                   for sc in scenes}
    consist = consistency.check(samples, kept_labels)
    if consist["violations"]:
        logger.error("跨任务一致性核对发现 %d 处冲突，同一张图配了两套真值，"
                     "详见 build_report.json 的 consistency 段：\n  %s",
                     len(consist["violations"]), "\n  ".join(consist["violations"][:5]))

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
        "vlm_endpoints": {e.name: {"model": e.model,
                                   "concurrency": e.concurrency,
                                   "calls": vlm.by_endpoint.get(e.name, 0),
                                   "fatal": e.fatal}
                          for e in vlm.endpoints},
        "phrase_bank": bank_stats,
        # 问句去重率：不同问法 / 问句总数。这个数掉下来就说明问法在复读，
        # 模型会把问句当固定口令背下来而不是听懂要框什么。
        "question_variety": _question_variety(samples),
        # 像素核对颜色拦下了多少条。dropped 明显偏高说明模型看颜色不准，
        # 该换个视觉能力更强的模型，或者把 attribute_qa 的配比调低。
        "color_check": {
            **color_stats,
            # 丢弃率高时（比如超过 20%）对着这几条打开原图看：
            # 是模型真看错了，还是这道闸对你的数据太严。
            # 后者就把 quality.verify_color_with_pixels 关掉。
            "rate": round(sum(v for k, v in color_stats.items()
                              if k.startswith("dropped"))
                          / max(1, color_stats["checked"]), 4),
            "examples": color_dropped[:15],
        },
        "consistency": {"checked_images": consist["checked_images"],
                        "violations": len(consist["violations"]),
                        "detail": consist["violations"][:50]},
        "split": _split_stats(train, val),
        "classes_total": table.count,
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["output"] = {"train": str(output_dir / "train.jsonl"),
                       "val": str(output_dir / "val.jsonl")}
    return stats



def _question_variety(samples) -> Dict[str, Any]:
    """统计人类问句的重复情况：去重后的问法数 / 问句总数，外加复现最多的那句。"""
    asked = [t["value"].replace("<image>\n", "")
             for s in samples for t in s["conversations"] if t["from"] == "human"]
    if not asked:
        return {}
    counts = Counter(asked)
    top, n = counts.most_common(1)[0]
    return {"distinct": len(counts), "total": len(asked),
            "ratio": round(len(counts) / len(asked), 4),
            "most_repeated": {"text": top[:40], "times": n}}


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



def _drop_bad_colors(ann, kept, vlm_info, stats, samples_dropped) -> int:
    """把和像素明显冲突的颜色说法从 vlm_info 里摘掉，返回核对了几个目标。

    只摘颜色，不整条丢：模型可能颜色看错但位置、类别都对，
    那这个目标换个非颜色属性（「车头朝左」）照样能用。
    """
    boxes = {b.index: b for b in kept}
    checked = 0
    for idx, info in vlm_info.items():
        b = boxes.get(idx)
        if b is None:
            continue
        px = [int((b.cx - b.w / 2) * ann.width), int((b.cy - b.h / 2) * ann.height),
              int((b.cx + b.w / 2) * ann.width), int((b.cy + b.h / 2) * ann.height)]
        hsv = None
        for field in ("color", "attribute"):
            claimed = info.get(field) or ""
            if colorcheck.color_word(claimed) is None:
                continue          # 不是颜色说法，这道闸不管
            if hsv is None:
                hsv = colorcheck.sample_hsv(ann.image_path, px)
                checked += 1
            if colorcheck.conflicts(claimed, hsv):
                info[field] = ""
                stats[f"dropped_{field}"] += 1
                # 留几条证据进报告。丢弃率高的时候光看数字判断不了是模型看错了
                # 还是这道闸太严 —— 得能对着原图看具体是哪个框、说的什么颜色、
                # 实测什么颜色。
                if len(samples_dropped) < 30:
                    samples_dropped.append({
                        "image": ann.image_path.name, "label": b.label,
                        "bbox_px": px, "field": field, "claimed": claimed,
                        "measured_hsv": [round(hsv[0]), round(hsv[1], 2),
                                         round(hsv[2], 2)],
                    })
    return checked


def _image_value(image_path: Path, style: str) -> str:
    """样本里 images 字段写什么。

    filename（默认）  裸文件名，LLaMA-Factory 按 media_dir 拼
    absolute          绝对路径，省掉配 media_dir 这一步
    relative          原样保留配置里给的相对路径，分隔符统一成正斜杠 ——
                      Windows 上生成、Linux 上训练时反斜杠会被当成转义符
    """
    if style == "absolute":
        return str(image_path.resolve())
    if style == "relative":
        return str(image_path).replace("\\", "/")
    return image_path.name


def _write_jsonl(path: Path, rows, include_meta: bool = True) -> None:
    """落盘。include_metadata=false 时把 metadata 摘掉 —— 训练用不到它，
    但构建全程要用（校验、一致性核对、配比统计），所以只在最后这一步删。"""
    # newline="\n"：Windows 上文本模式会把 \n 写成 \r\n，同一份配置在本地和
    # 服务器上跑出来的 jsonl 字节不一致，diff 和校验和都对不上。
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            if not include_meta:
                row = {k: v for k, v in row.items() if k != "metadata"}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

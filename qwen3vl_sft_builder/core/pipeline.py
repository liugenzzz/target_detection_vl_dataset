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
from .difficulty import (HARD, REJECT, Grader, balance_hard_quota, grade_at_most,
                          grade_rank)
from .grouping import source_group_key
from .referring import spatial_phrase
from .tasks import DESCRIBE_KINDS, MAIN_LINE, TASKS, Ctx


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
from . import colorcheck, consistency, describe_kinds, phrase_bank, progress
from .vlm_client import VlmClient
from .yolo import iter_annotations

logger = logging.getLogger(__name__)

# 一个样本槽位最多尝试几个任务类型。太小会浪费槽位，太大会让配比偏离权重。
MAX_TRY = 4



def _next_kind(kind_list, cursor, grades):
    """轮转表里下一个【这张图派得下去】的描述子类型，返回 (子类型, 新游标)。

    全是困难目标的图上派 `part`/`contrast`，模型只能返回空，白占一个槽位。
    一圈都没有合适的就退回轮转表里的那一个 —— 宁可让模型自己拒绝，
    也不能不给指派，那样描述又会坍缩回它最省力的那种。
    """
    k = kind_list[cursor % len(kind_list)]
    for _ in range(len(kind_list)):
        k = kind_list[cursor % len(kind_list)]
        cursor += 1
        if any(grade_at_most(g, k.max_grade) for g in grades):
            break
    return k, cursor


def _kinds_absent_here(vlm_info) -> set:
    """这张图上【不可能成立】的 ground_* 任务名。

    每个目标在预取阶段只被指派一种描述子类型，而调度器按缺口选任务时看不到
    这张图上有哪些子类型。一张图平均只有 1.7 个带描述的目标，却有 7 个
    ground_* 在抢 —— 十有八九点到这张图上不存在的那种，槽位直接作废。
    实测 100 张图有 169 个带描述的目标，主线只出了 70 条，99 个白白浪费。

    这【不是新增闸门】：没有对应子类型的目标，任务本来也必然返回 None。
    只是把「注定失败」提前算出来，省下的槽位留给能成的任务。
    描述或问句缺一不可 —— 主线要求三段齐全，缺一段一样出不来。
    """
    present = {info.get("describe_kind") for info in vlm_info.values()
               if info.get("description") and info.get("describe_q")}
    return {f"ground_{k}" for k in DESCRIBE_KINDS if k not in present}


def _deficit_order(target, made, failed_here, rng, strict=True):
    """按【产出缺口】排任务，欠得最多的在前；本图已证明出不了的排除掉。

    缺口 = 已产出 - 应产出 = made[t] - target[t] * (总产出 + 1)。越负欠得越多。

    【strict 时只返回还欠着的任务（缺口 < 0）】。这一条是配比的最后一道保险：
    一张图上主线的九个任务可能一个都成立不了（VLM 没挑中目标、或挑中的目标
    没被指派到当前欠账的那种描述子类型），这时如果照样把槽位填给「永远能成功」
    的非主线任务，它们就会把主线让出的槽位全部接走。
    实测过一次：主线 63.1% 掉到 34.9%，而 exist_negative / region_identify /
    detect_class / attribute_qa 四个全部超发到配比的 2.1~2.3 倍。
    宁可空过这个槽位 —— 数据量的瓶颈在有多少张图，配比歪了却补不回来。

    【failed_here 是必须的】。缺口只在 made 变化时才变，任务失败时 made 不变 ——
    排序结果一模一样，下一个槽位又选中同一个任务，再失败，再选中。strict 模式
    只试缺口最大的那一个，就此死锁。实测 500 张图：inventory_locate 试 384 次
    成功 1 次，之后锁死在 ground_unique 上失败 2515 次，另外 12 个任务一次都没
    被调用过，报告里连它们的失败计数都是 0，看不出任何异常。
    """
    done_all = sum(made.values())

    def gap(t):
        return made[t] - target[t] * (done_all + 1)

    order = sorted(target, key=lambda t: (gap(t), rng.random()))
    return [t for t in order
            if t not in failed_here and (not strict or gap(t) < 0)]


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
    # 互相打架的阈值会让某个开关静默失效，跑完全量才发现就太晚了。
    for conflict in grader.config_conflicts():
        logger.warning("配置冲突：%s", conflict)
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
    show_progress = bool(cfg.get_path("vlm.progress", True))
    kinds = describe_kinds.load_all()
    # 按 tasks 里各 ground_* 的权重排出轮转表 —— 权重大的在表里出现次数多，
    # 指派频率就跟着配比走。
    kind_list = [kinds[n] for n, w in sorted(_kind_weights(weights, kinds).items())
                 for _ in range(w) if n in kinds]
    if not kind_list:
        kind_list = list(kinds.values())
    kind_limits = describe_kinds.limits(kinds)
    kind_cursor = 0
    kind_plan: Dict[str, List[str]] = {}    # 图 -> [第1个挑中的子类型, 第2个, ...]
    kind_stats = Counter()
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
        # 【困难目标全局配额】。质量过滤只把「糊到没法用」的丢掉，剩下的里面
        # 困难档仍然很多 —— 实测不做配额时产出里困难目标占 43%，而要求是 10%。
        # 配额必须【全局】做：逐图做的话，一张全是困难目标的图仍会贡献满额困难
        # 样本，加起来还是超标（实测逐图配额得到 40.7%）。所以这里先记下候选，
        # 扫完全部图再统一下采样。
        raw_counts = Counter(b.label for b in ann.boxes)
        scenes.append({"ann": ann, "kept": kept, "gmap": gmap,
                       "raw_counts": dict(raw_counts)})
        if limit and len(scenes) >= limit:
            break

    # 【困难目标全局配额】。质量过滤只把「糊到没法用」的丢掉，剩下的里面困难档
    # 仍然很多 —— 实测不做配额时产出里困难目标占 43%，而要求是 10%。
    #
    # 配额必须【全局】做：逐图做的话，一张全是困难目标的图仍会贡献满额困难样本，
    # 加起来还是超标（实测逐图配额得到 40.7%）。所以扫完全部图再统一下采样。
    hard_quota = float(cfg.get_path("difficulty.hard_quota", 0.10))
    keys = [(si, b.index) for si, sc in enumerate(scenes) for b in sc["kept"]]
    kept_keys = set(balance_hard_quota(
        keys, lambda k: scenes[k[0]]["gmap"][k[1]].grade, hard_quota, seed))
    hard_before = sum(1 for k in keys if scenes[k[0]]["gmap"][k[1]].grade == HARD)
    for si, sc in enumerate(scenes):
        sc["kept"] = [b for b in sc["kept"] if (si, b.index) in kept_keys]
    scenes = [sc for sc in scenes if sc["kept"]]
    hard_after = sum(1 for sc in scenes for b in sc["kept"]
                     if sc["gmap"][b.index].grade == HARD)
    boxes_after = sum(len(sc["kept"]) for sc in scenes)
    quota_stats = {
        "hard_before": hard_before, "hard_after": hard_after,
        "boxes_after": boxes_after,
        "hard_ratio": round(hard_after / boxes_after, 4) if boxes_after else 0.0,
        "quota": hard_quota,
    }

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
            # 【按目标轮转指派描述子类型】。让模型自己挑会坍缩到最省力的那种
            # （实测过一次：给它「换动词换句式」的自由，它就只换同义词、
            # 骨架一个不动）。代码指派、模型可拒绝，分布才控得住。
            n_slots = min(max_pick, len(kept))
            grades_here = [sc["gmap"][b.index].grade for b in kept]
            slots, lines = [], []
            for i in range(n_slots):
                k, kind_cursor = _next_kind(kind_list, kind_cursor, grades_here)
                slots.append(k.name)
                lines.append(describe_kinds.render_assignment(k, i + 1))
            kind_plan[str(ann.image_path)] = slots
            text = prompts.render("vlm_select",
                                  box_list=_box_list_text(kept, bbox2d_for(ann)),
                                  max_pick=min(max_pick, len(kept)),
                                  kind_assignments="\n\n".join(lines))
            tasks.append((ann.image_path, [ann.width, ann.height], "scene", text))
        vlm.prefetch(tasks)

    # ---- 阶段三：按配比生成 ----
    samples: List[Dict[str, Any]] = []
    made = Counter()
    failed = Counter()
    invalid = 0

    vlm_cov: Counter = Counter()
    gen_bar = progress.make("生成样本", len(scenes), show_progress)
    for sc in scenes:
        ann, kept, gmap = sc["ann"], sc["kept"], sc["gmap"]
        b2d = bbox2d_for(ann)
        vlm_info = vlm.scene_info(ann.image_path, [ann.width, ann.height],
                                 {b.index for b in kept})
        # 【像素核对颜色】模型说「白色车身的卡车」而框落在蓝色卡车上时，
        # 指代就指向了错误的目标，attribute_qa 更是直接答错。颜色能从像素量出来，
        # 不该靠模型自觉。对不上就把这个颜色说法丢掉（其余字段照用），
        # 判据刻意宽松，只拦明显冲突。
        # 把指派的子类型贴回每个目标，并检查答案有没有跑出这个类型的范围 ——
        # 模型很容易把「只说外观」写成「位于画面左侧的一辆红色三轮车」，
        # 加了方位就又滑回三段式了。没有这道闸，七种跑几轮会退化成同一种。
        # 按【模型挑中的顺序】把子类型贴回去 —— scene_info 保序返回。
        slots = kind_plan.get(str(ann.image_path), [])
        for i, (bid, info) in enumerate(vlm_info.items()):
            kname = slots[i] if i < len(slots) else (slots[-1] if slots else "full")
            info["describe_kind"] = kname
            bad = describe_kinds.answer_violates(kinds[kname], info.get("description", ""))
            if bad:
                kind_stats[f"out_of_scope_{kname}"] += 1
                info["description"] = info["describe_q"] = ""
            elif info.get("description"):
                kind_stats[f"ok_{kname}"] += 1
            else:
                kind_stats[f"declined_{kname}"] += 1
        if color_gate:
            color_stats["checked"] += _drop_bad_colors(
                ann, kept, vlm_info, color_stats, color_dropped)
        # 【VLM 覆盖率】。主线九个任务都要求 VLM 挑中了目标并写出了描述，
        # 覆盖率低时它们会集体失败，而失败计数只会告诉你「出不了」，不会告诉你
        # 是模型没挑中、还是挑了但没写描述、还是任务自己的闸太严。
        vlm_cov["图片数"] += 1
        if vlm_info:
            vlm_cov["模型挑中过目标的图"] += 1
        vlm_cov["挑中的目标数"] += len(vlm_info)
        vlm_cov["其中写了描述的"] += sum(
            1 for v in vlm_info.values() if v.get("description"))
        vlm_cov["其中写了描述问句的"] += sum(
            1 for v in vlm_info.values() if v.get("describe_q"))
        vlm_cov["其中写了指代属性的"] += sum(
            1 for v in vlm_info.values() if v.get("attribute"))
        vlm_cov["过滤后的框数"] += len(kept)
        # ground_unique 要求「该类在原始标注里只有一个实例」——
        # 部件级标注下同类天然重复，这一条可能一张图都满足不了。
        vlm_cov["满足单实例的类别数"] += sum(
            1 for l, c in collections.Counter(b.label for b in kept).items()
            if c == 1 and sc["raw_counts"].get(l, c) == c)

        used: set = set()          # 本图已出过样本的框，避免同一目标反复出样本
        # 【本图已经证明出不了的任务】。缺口排序只在 made 变化时才会变，
        # 任务失败时 made 不变 —— 于是下一个槽位又选中同一个任务，再失败，
        # 再选中……strict 模式只试缺口最大的那一个，就此死锁在它身上。
        # 实测：500 张图里 inventory_locate 试了 384 次成功 1 次，之后锁死在
        # ground_unique 上失败 2515 次，另外 12 个任务【一次都没被调用过】。
        # 同一张图上同一个任务的输入是同一份，失败一次就不必再试。
        # 这个集合每张图重置，所以概率性失败的任务在下一张图还有机会。
        failed_here: set = set()
        # 【预先排除这张图上不可能成立的 ground_* 】。每个目标在预取阶段只被
        # 指派一种描述子类型，而调度器按缺口选任务时看不到这张图上有哪些子类型。
        # 一张图平均只有 1.7 个带描述的目标，却有 7 个 ground_* 在抢 ——
        # 十有八九点到这张图上不存在的那种，槽位直接作废。
        # 实测：100 张图有 169 个带描述的目标，主线只出了 70 条，99 个白白浪费。
        # 这里不是新增闸门，是把「注定失败」提前算出来，省下的槽位留给能成的。
        failed_here |= _kinds_absent_here(vlm_info)
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
            deficit = _deficit_order(target, made, failed_here, rng,
                                     strict=strict_ratio)
            if not deficit:
                break                    # 这张图上所有任务都试过了，别空转

            def make_ctx():
                return Ctx(annotation=ann, boxes=kept, grades=gmap, vlm=vlm_info,
                           all_labels=all_labels,
                           bbox2d=b2d, spatial=lambda b: spatial_phrase(b.cx, b.cy),
                           rng=rng, short_answer=rng.random() < short_ratio,
                           measure_words=measure_words, require_desc=require_desc,
                           used=used, min_desc_len=min_desc_len,
                           raw_counts=sc["raw_counts"], forbid_chat=forbid_chat,
                           confusable=confusable, hypernym=hypernym,
                           kind_limits=kind_limits, json_fence=json_fence)

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
                failed_here.add(name)
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
                    # 评估报表要按难度档和尺寸档拆开看。困难目标只占 10%，
                    # 不能拆就会被 90% 的简单目标稀释成「还行」。
                    # 多框任务取【最难/最小】的那个框 —— 一条样本的难度由
                    # 它最难的那个目标决定，取平均会把小目标抹平。
                    **_focus_difficulty(out.get("focus", []), gmap, grader),
                    "n_turns": len(out["conversations"]) // 2,
                    **{k: v for k, v in out.items()
                       if k in ("attribute", "attribute_kind", "describe_kind",
                                "relation",
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
        gen_bar.step(note=f"{len(samples)} 条")
    gen_bar.close()

    # ---- 阶段四：按来源分组划分 ----
    train, val, test = _split_by_source(samples, cfg, seed)
    _write_jsonl(output_dir / "train.jsonl", train, include_meta)
    _write_jsonl(output_dir / "val.jsonl", val, include_meta)
    # test 永远带 metadata：评估报表要按 task_type / difficulty / size_bucket
    # 拆分，摘掉就拆不了。它不进训练，多这几个字段没有代价。
    _write_jsonl(output_dir / "test.jsonl", test, True)

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
        # 困难目标下采样前后。hard_ratio 应该贴近 quota ——
        # 差很多说明简单档本身就不够，配额算式被 min() 截断了。
        "hard_quota": quota_stats,
        "samples_total": sum(made.values()),
        "by_task_type": {k: made[k] for k in TASKS if made[k]},
        "task_ratio_actual": {k: round(made[k] / total, 4) for k in TASKS if made[k]},
        "main_line_ratio": round(sum(made[k] for k in MAIN_LINE) / total, 4),
        "short_answer_ratio_actual": round(
            sum(1 for s in samples if s["metadata"]["answer_format"] == "short") / total, 4),
        # 样本层面的难度/尺寸分布。上面的 hard_quota 数的是【框】，这里数的是
        # 【样本】—— 一个困难框可能只出一条样本，两个数不必相等。评估报表按这
        # 两根轴拆分，先在构建期确认每一格都有足够的量，否则评估时会出现
        # n=3 的格子。
        "difficulty_mix": _meta_mix(samples, "difficulty"),
        "size_mix": _meta_mix(samples, "size_bucket"),
        # 主线九个任务都要求 VLM 挑中目标并写出描述。这一段区分三种失败：
        # 模型没挑中 / 挑了但没写描述 / 写了但任务自己的闸拦下了。
        # 只看 task_unavailable 分不出来，会往错误方向调。
        "vlm_coverage": dict(vlm_cov),
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
        # 每种描述子类型：模型接了多少、拒绝了多少、写跑题被丢了多少。
        # declined 高说明这个子类型对你的数据不适用（航拍小目标看不清部件），
        # out_of_scope 高说明 prompts/describe/<子类型>.txt 的要求还不够硬。
        "describe_kinds": dict(kind_stats),
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
        "split": _split_stats(train, val, test),
        "classes_total": table.count,
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["output"] = {"train": str(output_dir / "train.jsonl"),
                       "val": str(output_dir / "val.jsonl"),
                       "test": str(output_dir / "test.jsonl")}
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


def _meta_mix(samples, key: str) -> Dict[str, Any]:
    """某个 metadata 字段的取值分布，含占比。None 归到 "none"（拒答类没有焦点框）。"""
    counts = Counter(str(s["metadata"].get(key)) for s in samples)
    total = sum(counts.values()) or 1
    return {"counts": dict(counts),
            "ratio": {k: round(v / total, 4) for k, v in counts.items()}}


def _focus_difficulty(focus, gmap, grader) -> Dict[str, Any]:
    """这条样本聚焦的框有多难、有多大。没有聚焦框（拒答类）时全为 None。"""
    grades = [gmap[i] for i in focus if i in gmap]
    if not grades:
        return {"difficulty": None, "area_ratio": None,
                "equiv_px": None, "size_bucket": None}
    g = min(grades, key=lambda x: x.area_ratio)     # 最小的那个说了算
    hardest = max(grades, key=lambda x: grade_rank(x.grade))
    return {"difficulty": hardest.grade,
            "area_ratio": round(g.area_ratio, 6),
            "equiv_px": round(g.equiv_px, 1),
            "size_bucket": grader.bucket_of(g)}


def _split_by_source(samples, cfg, seed) -> Tuple[List, List, List]:
    """按【原始来源】分组划分。数据含视频抽帧与 Roboflow 增强，
    按图片随机划分会让同源图泄漏进验证集，指标虚高且难以察觉。

    切三路而不是两路：val 会被训练框架拿去算 eval loss、挑 checkpoint，
    等于参与了调参；拿它报模型成绩是在自己给自己出卷子。test 全程不进
    训练流程，只在评估时拆封。
    """
    val_ratio = float(cfg.get_path("split.val_ratio", 0.05))
    test_ratio = float(cfg.get_path("split.test_ratio", 0.05))
    groups: Dict[str, List] = {}
    for s in samples:
        groups.setdefault(source_group_key(s["images"][0]), []).append(s)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(samples)
    test: List = []
    val: List = []
    train: List = []
    # 先填 test 再填 val：test 是要拿去报成绩的，宁可 val 少一点。
    for k in keys:
        if len(test) < n * test_ratio:
            test.extend(groups[k])
        elif len(val) < n * val_ratio:
            val.extend(groups[k])
        else:
            train.extend(groups[k])
    return train, val, test


def _split_stats(train, val, test) -> Dict[str, Any]:
    def groups(rows):
        return {source_group_key(s["images"][0]) for s in rows}

    tg, vg, sg = groups(train), groups(val), groups(test)
    return {"train": len(train), "val": len(val), "test": len(test),
            "train_groups": len(tg), "val_groups": len(vg), "test_groups": len(sg),
            # 三个都必须是 0。非 0 说明同源图跨了划分，评估数字不可信。
            "group_overlap": len(tg & vg) + len(tg & sg) + len(vg & sg),
            "overlap_train_val": len(tg & vg),
            "overlap_train_test": len(tg & sg),
            "overlap_val_test": len(vg & sg)}



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



# 描述子类型的任务名统一是 ground_<子类型>，这样配比表里一眼看得出是同一族。
KIND_TASK_PREFIX = "ground_"


def _kind_weights(weights, kinds) -> Dict[str, int]:
    """从 tasks 配比里抽出各描述子类型的权重。"""
    out = {}
    for name in kinds:
        w = int(weights.get(KIND_TASK_PREFIX + name, 0) or 0)
        if w > 0:
            out[name] = w
    return out


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

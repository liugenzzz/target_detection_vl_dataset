"""十种任务的样本生成器。

每个任务一个函数，签名统一为 (ctx) -> dict | None，返回 None 表示这张图上
出不了这个任务（条件不满足）。注册在 TASKS 里，配比由 config 的 tasks 段控制，
把某个任务的权重设为 0 即可关闭它。

主线是 ground_unique / ground_spatial / ground_attribute 三个，合计 45%，
即「指代物体 -> 框 -> 描述语句」。拆成三个是为了能分别筛选、分别核算成本 ——
前两个纯模板零成本，第三个要调 VLM 拿属性。

任务体系参考了几个公开数据集的实际拆分：
  RefCOCO/+/g   指代理解（REC），指代带类别名是正确的，答案是框不是类别
  Ferret GRIT   四类：单物体 / 物体关系 / 区域描述 / 区域推理，含 95K hard negative
  Osprey-724K   属性是最大一类（207K，29%），另有 14% 的短答案格式样本
  Kosmos-2      区域识别（REG）的 prompt 形如 It <box>...</box> is
  COCO-QA       四类：Object / Number / Color / Location，答案是单词或短语
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import prompts

# 属性问答目前只问颜色。VLM 返回的 attribute 是「用于指认的最显眼特征」，
# 可能是朝向、形状，不一定是颜色，所以两者分开存。
ATTR_QUESTIONS = {"color": "颜色"}


@dataclass
class Ctx:
    """生成一条样本所需的全部上下文。"""
    annotation: Any                      # core.yolo.Annotation
    boxes: List[Any]                     # 通过质量过滤的框
    grades: Dict[int, Any]               # box_index -> Grade
    vlm: Dict[int, Dict[str, str]]       # box_index -> {attribute,color,description}
    clean_labels: set                    # 该图中「全部框都合格」的类别，可出 detect/count
    all_labels: List[str]                # 全类别表，用于挑不存在的类别做拒答
    bbox2d: Callable                     # box -> [x1,y1,x2,y2]
    spatial: Callable                    # box -> 「下方左侧」
    rng: random.Random
    short_answer: bool = False
    measure_words: Dict[str, str] = field(default_factory=dict)
    require_desc: bool = True            # 主线是否强制三段齐全
    # 本张图已经被出过样本的框。同一个目标出多条样本时，答案 bbox 完全相同，
    # 只是问法不同，属于近重复数据 —— 实测同一个框曾在一张图里被出了 4 次。
    used: set = field(default_factory=set)

    def unused(self, boxes):
        """过滤掉本图已用过的框；全都用过时返回空列表，任务据此跳过。"""
        return [b for b in boxes if b.index not in self.used]

    def mw(self, label: str) -> str:
        """该类别的量词。船「艘」、车「辆」、人「名」；查不到退回「个」。"""
        return self.measure_words.get(label, "个")


def _j(v) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _box_of(ctx: Ctx, b) -> Dict[str, Any]:
    return {"bbox_2d": ctx.bbox2d(b), "label": b.label}


def _turns(*pairs) -> List[Dict[str, str]]:
    out = []
    for i, (q, a) in enumerate(pairs):
        out.append({"from": "human", "value": (f"<image>\n{q}" if i == 0 else q)})
        out.append({"from": "gpt", "value": a})
    return out


# 答案是 bbox JSON 的任务。「用一个词回答」对它们没有意义 ——
# 实测会生成出「定位图中银灰色的那辆遮阳三轮车。用一个词回答。」而答案是一段 JSON。
_BOX_ANSWER_TASKS = {"ground_unique", "ground_spatial", "ground_attribute", "detect_class"}


def _ask(ctx: Ctx, text: str, task: str = "") -> str:
    """需要短答案时，在问题后追加格式要求。只对文本答案的任务生效。"""
    if ctx.short_answer and task not in _BOX_ANSWER_TASKS:
        return f"{text}{prompts.load('short_answer_suffix')}"
    return text


def _clean_attr(attr: str) -> str:
    """去掉属性词尾部的「的」。模板与 VLM 问句都写成「{attribute}的{label}」，
    模型返回「蹲着的」时会拼出「蹲着的的行人」。"""
    return (attr or "").strip().rstrip("的") or (attr or "").strip()


def _same_label(ctx: Ctx, label: str) -> List:
    return [b for b in ctx.boxes if b.label == label]


def _describe(ctx: Ctx, b) -> Optional[str]:
    d = (ctx.vlm.get(b.index) or {}).get("description")
    return d or None


# --------------------------------------------------------------- 主线三种
def ground_unique(ctx: Ctx):
    """定位图中的卡车。—— 该类在图中仅一个实例，类别名本身就唯一，无需任何修饰。"""
    singles = ctx.unused([b for b in ctx.boxes if len(_same_label(ctx, b.label)) == 1])
    if not singles:
        return None
    b = ctx.rng.choice(singles)
    return _main_line(ctx, b, prompts.render("ground_unique", label=b.label))


def _main_line(ctx: Ctx, b, question: str, extra=None):
    """组装一条主线样本：指代 -> 框 -> 描述。

    require_desc 开启时，拿不到描述就返回 None —— 主线的定义就是三段齐全，
    只有「指代 -> 框」不算主线。
    """
    desc = _describe(ctx, b)
    if ctx.require_desc and not desc:
        return None
    pairs = [(_ask(ctx, question, "ground_unique"), _j(_box_of(ctx, b)))]
    if desc:
        pairs.append((prompts.load("describe_target"), desc))
    out = {"conversations": _turns(*pairs), "focus": [b.index], "label": b.label}
    if extra:
        out.update(extra)
    return out


def ground_spatial(ctx: Ctx):
    """定位图中左侧那个人员。—— 空间指代，纯模板，零成本。

    要求该目标在 3x3 分区内同类唯一，否则指代锁不住。
    """
    cand = ctx.unused([b for b in ctx.boxes
                       if ctx.grades[b.index].unique_in_zone
                       and len(_same_label(ctx, b.label)) > 1])
    if not cand:
        return None
    b = ctx.rng.choice(cand)
    return _main_line(ctx, b, prompts.render(
        "ground_spatial", spatial=ctx.spatial(b), mw=ctx.mw(b.label), label=b.label))


def ground_attribute(ctx: Ctx):
    """属性指代定位 —— 主线。问句由 VLM 生成，不套模板。

    模板问句「定位图中{attribute}的那{mw}{label}。」写死后所有样本一个腔调，
    十万条同一句式，既不像真人说话，也让模型只学到那一个句式。改由 VLM 为每个
    目标生成三种说法，这里随机取一句；VLM 没给问句时才退回模板。
    """
    cand = ctx.unused([b for b in ctx.boxes
                       if (ctx.vlm.get(b.index) or {}).get("attribute")])
    if not cand:
        return None
    b = ctx.rng.choice(cand)
    info = ctx.vlm[b.index]
    attr = _clean_attr(info["attribute"])

    questions = [q.replace("的的", "的") for q in (info.get("questions") or []) if q]
    if questions:
        question = ctx.rng.choice(questions)
        source = "vlm"
    else:
        question = prompts.render("ground_attribute", attribute=attr,
                                  mw=ctx.mw(b.label), label=b.label)
        source = "template"
    return _main_line(ctx, b, question,
                      {"attribute": attr, "question_source": source})


# --------------------------------------------------------------- 其余七种
def detect_class(ctx: Ctx):
    """定位图中所有的人员。—— 答案必须完整，所以只在该类【全部框都合格】时才出。

    否则漏掉被质量过滤的框，等于在教模型漏检。
    """
    cand = [l for l in ctx.clean_labels if len(_same_label(ctx, l)) >= 2]
    if not cand:
        return None
    label = ctx.rng.choice(sorted(cand))
    boxes = _same_label(ctx, label)
    return {"conversations": _turns(
                (_ask(ctx, prompts.render("detect_class", label=label), "detect_class"),
                 _j([_box_of(ctx, b) for b in boxes]))),
            "focus": [b.index for b in boxes], "label": label}


def attribute_qa(ctx: Ctx):
    """图中 [框] 这个目标是什么颜色？—— Osprey-724K 里属性是最大一类（29%）。"""
    cand = ctx.unused([b for b in ctx.boxes if (ctx.vlm.get(b.index) or {}).get("color")])
    if not cand:
        return None
    b = ctx.rng.choice(cand)
    key = ctx.rng.choice(sorted(ATTR_QUESTIONS))
    color = ctx.vlm[b.index][key]
    return {"conversations": _turns(
                (_ask(ctx, prompts.render("attribute_qa", bbox=_j(ctx.bbox2d(b)),
                                          attr_name=ATTR_QUESTIONS[key])),
                 color if ctx.short_answer else f"是{color}的。")),
            "focus": [b.index], "label": b.label, "attribute": color}


def spatial_relation(ctx: Ctx):
    """图中的卡车在人员的哪一侧？—— 纯坐标计算，零 VLM 成本（Ferret 四类之一）。

    只取关系明确的一对：一个轴拉开、另一个轴接近，否则「左边」这种回答不成立。

    参照物 B 必须是该类在图中的唯一实例，否则「在卡车的哪一侧」指不明确。
    主体 A 只要在 3x3 分区内同类唯一即可，问句里带上空间修饰来消歧 ——
    最初要求 A 也是唯一实例，在密集数据上一条都出不来（实测 VisDrone 命中 0 次）。
    """
    singles = {l for l in {b.label for b in ctx.boxes} if len(_same_label(ctx, l)) == 1}
    if not singles:
        return None

    pool = []
    for a in ctx.boxes:
        if not ctx.grades[a.index].unique_in_zone:
            continue
        for b in ctx.boxes:
            if a.index == b.index or a.label == b.label or b.label not in singles:
                continue
            dx, dy = a.cx - b.cx, a.cy - b.cy
            if abs(dx) > 0.12 and abs(dy) < 0.12:
                pool.append((a, b, "右侧" if dx > 0 else "左侧"))
            elif abs(dy) > 0.12 and abs(dx) < 0.12:
                pool.append((a, b, "下方" if dy > 0 else "上方"))
    if not pool:
        return None

    a, b, rel = ctx.rng.choice(pool)
    # A 的类别有多个实例时，问句要带空间修饰才指得明确
    subject = (a.label if len(_same_label(ctx, a.label)) == 1
               else f"{ctx.spatial(a)}那{ctx.mw(a.label)}{a.label}")
    return {"conversations": _turns(
                (_ask(ctx, prompts.render("spatial_relation",
                                          label_a=subject, label_b=b.label)),
                 rel if ctx.short_answer else f"在{b.label}的{rel}。")),
            "focus": [a.index, b.index], "label": a.label, "relation": rel}


def count(ctx: Ctx):
    """图中有几个人员？—— 同样只在该类全部框合格时才出，否则数字是错的。"""
    if not ctx.clean_labels:
        return None
    label = ctx.rng.choice(sorted(ctx.clean_labels))
    n = len(_same_label(ctx, label))
    return {"conversations": _turns(
                (_ask(ctx, prompts.render("count", label=label)),
                 str(n) if ctx.short_answer
                 else prompts.render("count_answer", n=n, mw=ctx.mw(label)))),
            "focus": [], "label": label, "count": n}


def exist_negative(ctx: Ctx):
    """图中有没有直升机？—— 不可省。没有这类样本，模型会学到「被问就一定有」，
    推理时凭空编框。Ferret 有 95K hard negative（8.6%），Osprey 有 64K（8.8%）。

    一半问存在的类别（答有），一半问不存在的（答没有），避免模型学成「一律答没有」。
    """
    present = {b.label for b in ctx.boxes}
    if ctx.rng.random() < 0.5 and present:
        label = ctx.rng.choice(sorted(present))
        n = len(_same_label(ctx, label))
        q = prompts.render("exist_yes", label=label)
        a = "有" if ctx.short_answer else prompts.render(
            "exist_yes_answer", n=n, mw=ctx.mw(label), label=label)
        polarity = "positive"
    else:
        absent = [l for l in ctx.all_labels if l not in present]
        if not absent:
            return None
        label = ctx.rng.choice(absent)
        q = prompts.render("exist_no", label=label)
        a = "没有" if ctx.short_answer else prompts.render("exist_no_answer", label=label)
        polarity = "negative"
    return {"conversations": _turns((_ask(ctx, q), a)),
            "focus": [], "label": label, "polarity": polarity}


def region_identify(ctx: Ctx):
    """图中 [框] 这个位置的是什么？—— REG 方向，用坐标指向目标。

    没有文字指代，因此不存在任何「把答案写进问题」的可能。
    """
    cand = ctx.unused(ctx.boxes)
    if not cand:
        return None
    b = ctx.rng.choice(cand)
    a = b.label if ctx.short_answer else prompts.render("region_identify_answer", label=b.label)
    return {"conversations": _turns(
                (_ask(ctx, prompts.render("region_identify", bbox=_j(ctx.bbox2d(b)))), a)),
            "focus": [b.index], "label": b.label}


def image_caption(ctx: Ctx):
    """描述一下这张图片。—— 由各目标的 VLM 描述拼合，不再单独调用。"""
    seen, descs = set(), []
    for b in ctx.boxes:
        d = _describe(ctx, b)
        if d and d not in seen:          # 去重：多个目标拿到同一句描述时只留一句
            seen.add(d)
            descs.append(d)
    if len(descs) < 2:                   # 只有一句（或没有）拼不成整图描述
        return None
    return {"conversations": _turns(
                (prompts.load("image_caption"), "".join(descs[:4]))),
            "focus": [b.index for b in ctx.boxes], "label": None}


TASKS: Dict[str, Callable[[Ctx], Optional[Dict[str, Any]]]] = {
    "ground_unique": ground_unique,
    "ground_spatial": ground_spatial,
    "ground_attribute": ground_attribute,
    "detect_class": detect_class,
    "attribute_qa": attribute_qa,
    "spatial_relation": spatial_relation,
    "count": count,
    "exist_negative": exist_negative,
    "region_identify": region_identify,
    "image_caption": image_caption,
}

MAIN_LINE = ("ground_unique", "ground_spatial", "ground_attribute")

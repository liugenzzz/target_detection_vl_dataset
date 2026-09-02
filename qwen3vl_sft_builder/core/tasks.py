"""十四个任务的样本生成器。

每个任务一个函数，签名统一为 (ctx) -> dict | None，返回 None 表示这张图上
出不了这个任务（条件不满足）。注册在 TASKS 里，配比由 config 的 tasks 段控制，
把某个任务的权重设为 0 即可关闭它。

主线合计 70%，即「指代物体 -> 框 -> 描述语句」，由九个任务组成：
ground_unique、inventory_locate，以及七个 ground_<描述子类型>。

七个 ground_* 是【同一条链】的变体，只有最后那轮描述不同 —— 拆开是因为
描述的多样性要来自「答案里装的是什么」（只说本体 / 只说方位 / 只说邻接关系
……），不是问句换说法。各自的提示词在 prompts/describe/<子类型>.txt。

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

from . import register
from .difficulty import grade_at_most
from .referring import is_vacuous_description

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
    # 过滤后的框【就是】这张图的真值：所有任务都基于同一个集合，
    # 这样「图中有 3 名人员」和「定位所有人员 -> 3 个框」不会自相矛盾。
    # 曾经用「该类全部框都合格」来决定能否出盘点/检测类任务，结果同一张图上
    # 一个样本说没有人员、另一个样本又去定位人员，上下文打架。
    all_labels: List[str]                # 全类别表，用于挑不存在的类别做拒答
    bbox2d: Callable                     # box -> [x1,y1,x2,y2]
    spatial: Callable                    # box -> 「下方左侧」
    rng: random.Random
    short_answer: bool = False
    measure_words: Dict[str, str] = field(default_factory=dict)
    require_desc: bool = True            # 主线是否强制三段齐全
    min_desc_len: int = 18               # 描述短于此判为空话
    skeletons: Dict[int, Any] = field(default_factory=dict)   # box_index -> Skeleton
    # 【过滤前】各类别的框数。boxes 是过滤后的，两者的差就是被质量过滤掉的目标。
    # 穷举式问句（「定位图中的卡车」「框出图中所有的人员」）必须拿这个数来把关：
    # 原图有 4 辆三轮车、3 辆因太小被过滤，仍去问「定位图中的三轮车」并只给 1 个框，
    # 就是在教模型漏检。实测这种情况占 ground_unique 可选组合的 45.2%。
    raw_counts: Dict[str, int] = field(default_factory=dict)
    # 语体禁用词（config 的 phrase_banks.forbid_global）。ground_attribute 的问句
    # 是 VLM 按图现场生成的，不走问法库，那道闸管不到它 —— 这里现场过一遍。
    forbid_chat: tuple = ()
    # 坐标【答案】要不要用 ```json 包起来。问句里出现的坐标不受影响 ——
    # 那是指向目标的记号，不是模型要输出的东西。
    json_fence: bool = False
    # {类别名: [易混的类别名]}。拒答样本挑「图里有卡车，问有没有货车」这种，
    # 比从 347 类里随机抽一个不相干的东西有价值得多。
    confusable: Dict[str, List[str]] = field(default_factory=dict)
    # {类别名: [与它互为上下位的类别名]}。拒答样本必须避开，见 hypernyms_of。
    hypernym: Dict[str, List[str]] = field(default_factory=dict)
    # {描述子类型: 最难档位}。见 prompts/describe/<子类型>.txt 的 `#! max-grade:`。
    # 指派阶段已经按这张图的档位跳过了不适用的子类型，这里是兜底：
    # 缓存里可能还留着改限制之前派下去的结果。
    kind_limits: Dict[str, str] = field(default_factory=dict)
    # 本张图已经被出过样本的框。同一个目标出多条样本时，答案 bbox 完全相同，
    # 只是问法不同，属于近重复数据 —— 实测同一个框曾在一张图里被出了 4 次。
    used: set = field(default_factory=set)

    def kind_fits(self, kind: str, box) -> bool:
        """这个框的难度档位配不配做这一种描述。没配限制或没有档位信息就放行。"""
        limit = self.kind_limits.get(kind, "")
        if not limit:
            return True
        g = self.grades.get(box.index)
        return g is None or grade_at_most(g.grade, limit)

    def unused(self, boxes):
        """过滤掉本图已用过的框；全都用过时返回空列表，任务据此跳过。"""
        return [b for b in boxes if b.index not in self.used]

    def all_kept(self, label: str) -> bool:
        """该类别过滤前后的框数是否一致 —— 即没有任何一个实例被质量过滤掉。

        没传 raw_counts 时（单测直接构造 Ctx）退回「按没被过滤处理」。
        """
        kept = sum(1 for b in self.boxes if b.label == label)
        return self.raw_counts.get(label, kept) == kept

    def confusable_with(self, labels) -> List[str]:
        """跟这些类别容易混、但【不是】上下位关系的类别名。用于挑 hard negative。"""
        out = []
        for l in labels:
            out += self.confusable.get(l, [])
        return sorted(set(out))

    def hypernyms_of(self, labels) -> set:
        """与这些类别互为上下位的类别名（名字互相包含）。

        拒答样本必须避开它们：图里有遮阳三轮车，问「有没有三轮车」答「没有」
        是错的 —— 遮阳三轮车本来就是三轮车。
        """
        out = set()
        for l in labels:
            out |= set(self.hypernym.get(l, ()))
        return out

    def mw(self, label: str) -> str:
        """该类别的量词。船「艘」、车「辆」、人「名」；查不到退回「个」。"""
        return self.measure_words.get(label, "个")


# 坐标答案要不要用 ```json 代码块包起来。
# Qwen3-VL 基座模型自带这个习惯 —— 直接问它「框出图中全部的轿车」，
# 返回的是 ```json\n[...]\n```。训练数据用裸 JSON 会把这个习惯覆盖掉，
# 用代码块则是顺着基座的既有行为练。两种都能用，取决于下游解析器：
#   false（默认）  裸 JSON，省 token，解析器直接 json.loads 即可
#   true           带 ```json 包裹，与基座输出一致，已有解析器不用改
# 由 config 的 output.wrap_json_in_code_block 控制。
JSON_FENCE = "```json\n{body}\n```"


def _j(v, fence: bool = False) -> str:
    body = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    return JSON_FENCE.format(body=body) if fence else body


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


def _ask(ctx: Ctx, text: str, box_answer: bool = False) -> str:
    """需要短答案时，在问题后追加格式要求。

    box_answer=True 的问句答案是一串 JSON 坐标，给它接一句「用一个词回答」
    自相矛盾 —— 曾经真的产出过「定位图中银灰色的那辆三轮车。用一个词回答。」
    配一个 bbox_2d 的样本。
    """
    if ctx.short_answer and not box_answer:
        return f"{text}{prompts.load('short_answer_suffix')}"
    return text


def _clean_attr(attr: str) -> str:
    """去掉属性词尾部的「的」。模板与 VLM 问句都写成「{attribute}的{label}」，
    模型返回「蹲着的」时会拼出「蹲着的的行人」。"""
    return (attr or "").strip().rstrip("的") or (attr or "").strip()


def _norm_attr(attr: str) -> str:
    """属性归一化，用来判两个属性是不是「同一个说法」。

    去掉不承载区分信息的虚词 —— 「穿白色上衣」和「白色上衣」指的是同一件事，
    比字面相等要严，否则模型换个说法就绕过唯一性检查了。
    """
    out = _clean_attr(attr)
    for w in ("穿着", "穿", "戴着", "戴", "身着", "正在", "正", "着", "的"):
        out = out.replace(w, "")
    return out.strip("，,、 ")


def _attr_identifies(ctx: Ctx, b, attr: str) -> bool:
    """这个属性能不能在【同类目标】里唯一指认 b。

    实测踩到：一张公园图里好几个人都穿白上衣，VLM 给其中两个都写了
    「穿白色上衣」，于是生成了两条问句几乎一样、答案却是不同框的样本 ——
    同一个问题两个正确答案，模型只能学成随机猜。

    包含关系也算不唯一：「白色上衣」和「白色上衣、黑色裤子」这两个说法，
    前者同时匹配两个人，照样指不明确。
    """
    key = _norm_attr(attr)
    if not key:
        return False
    for o in _same_label(ctx, b.label):
        if o.index == b.index:
            continue
        other = _norm_attr((ctx.vlm.get(o.index) or {}).get("attribute", ""))
        if not other:
            continue
        if key == other or key in other or other in key:
            return False
    return True


def _same_label(ctx: Ctx, label: str) -> List:
    return [b for b in ctx.boxes if b.label == label]


def _describe(ctx: Ctx, b) -> Optional[str]:
    """取该目标的描述。空话描述当作没有 —— 主线要求三段齐全时会因此跳过该目标，
    宁可少一条样本，也不要「画面中一处目标，轮廓清晰」这种照着找不到的描述。"""
    d = (ctx.vlm.get(b.index) or {}).get("description")
    if not d or is_vacuous_description(d, ctx.min_desc_len):
        return None
    return d


# --------------------------------------------------------------- 主线三种
def ground_unique(ctx: Ctx):
    """定位图中的卡车。—— 该类在图中仅一个实例，类别名本身就唯一，无需任何修饰。

    「仅一个实例」按【原始标注】算，不是按过滤后算。原图有 4 辆三轮车、
    3 辆被质量过滤掉，问「定位图中的三轮车」再只给 1 个框，是在教模型漏检。
    """
    singles = ctx.unused([b for b in ctx.boxes
                          if len(_same_label(ctx, b.label)) == 1
                          and ctx.all_kept(b.label)])
    if not singles:
        return None
    b = ctx.rng.choice(singles)
    return _main_line(ctx, b, prompts.render_choice("ground_unique", ctx.rng,
                                                    label=b.label))


def _box_question_ok(ctx: Ctx, q: str) -> bool:
    """VLM 现场写的定位问句合不合格。

    过的是 ground_attribute 问法池自己那套规则 —— 手写种子过得了的，
    模型写的也得过。之前这里只查了语体，漏了「必须明说要框」那一条，
    于是模型写的「白色面包车在图中的什么位置？」一路进了数据，
    而它的答案是一串坐标。实测真模型问这句时答的是一段方位描述，
    不是框 —— 同一个问句两种答案，正是我们要避免的。
    """
    if not register.is_instruction(q, ctx.forbid_chat):
        return False
    if any(w in q for w in prompts.forbidden_of("ground_attribute")):
        return False
    req = prompts.required_any_of("ground_attribute")
    return not req or any(w in q for w in req)


def _describe_question(ctx: Ctx, b) -> str:
    """第三轮「描述这个目标」的问句。

    优先用 VLM 和 description 成对生成的那一句 —— 同一次调用里写出来的，
    问句问了哪几样、答句就答哪几样，对得上。模板池是各写各的，
    问「介绍一下这辆自行车」而答颜色方位这种问答脱节就是这么来的。

    模型没给、或给的那句自己就不合格（宽泛、跑去问坐标）时回落模板池。
    """
    q = (ctx.vlm.get(b.index) or {}).get("describe_q", "")
    if q and _describe_q_ok(ctx, q):
        return q
    return prompts.render_choice("ask_describe", ctx.rng,
                                 mw=ctx.mw(b.label), label=b.label)


# 描述问句必须点明要什么。只禁用词不够 —— 「这辆车在哪？」一个禁用词都不沾，
# 但它宽泛得答什么都算对。答案里有外观、方位、周边三样，问句至少要点到
# 外观和方位，否则问的和答的对不上。
_WANTS_LOOK = ("外观", "外形", "什么样", "样子", "颜色", "特征", "样式")
_WANTS_WHERE = ("方位", "位置", "周围", "旁边", "周边", "附近", "环境", "画面")


def _describe_q_ok(ctx: Ctx, q: str) -> bool:
    """模型写的描述问句合不合格。用问法池自己那套闸来判 ——
    手写种子过得了的规则，模型写的也得过。"""
    if not register.is_instruction(q, ctx.forbid_chat):
        return False
    if any(w in q for w in prompts.forbidden_of("ask_describe")):
        return False
    if not (any(w in q for w in _WANTS_LOOK) and any(w in q for w in _WANTS_WHERE)):
        return False
    return 8 <= len(q) <= prompts.max_len_of("ask_describe", 45)


def _main_line(ctx: Ctx, b, question: str, extra=None):
    """组装一条主线样本：指代 -> 框 -> 描述。

    require_desc 开启时，拿不到描述就返回 None —— 主线的定义就是三段齐全，
    只有「指代 -> 框」不算主线。
    """
    desc = _describe(ctx, b)
    if ctx.require_desc and not desc:
        return None
    pairs = [(_ask(ctx, question, box_answer=True), _j(_box_of(ctx, b), ctx.json_fence))]
    if desc:
        pairs.append((_describe_question(ctx, b), desc))
    out = {"conversations": _turns(*pairs), "focus": [b.index], "label": b.label}
    if extra:
        out.update(extra)
    return out


def inventory_locate(ctx: Ctx):
    """盘点 -> 定位 -> 描述，三轮递进 —— 主线。

        用户 │ 图中有哪些清晰可见的目标？
        模型 │ 有 3 名人员、2 辆卡车和 1 艘船。
        用户 │ 那艘船在哪？
        模型 │ {"bbox_2d": [...], "label": "其它辅助船"}
        用户 │ 描述一下这艘船。
        模型 │ ...

    每一轮都承接上一轮，是一段真对话，不是拼起来的问答对。计数信息在第一轮
    自然带出，所以不再单独出「图中有几个 X」的任务。

    第一轮的问法刻意限定为【清晰可见的】目标：实测只有 3.5% 的图片全部标注框
    都能通过质量过滤，其余图片都有被剔除的小目标。不加这个限定就等于给出一份
    不完整的清单，是在教模型漏报。加了限定则与整套质量过滤的立场自洽 ——
    小目标本来就不进训练集。

    第二轮要挑一个【该类只有一个干净实例】的目标，这样「那艘船」无歧义。
    """
    # 只列全部框都合格的类别，被过滤掉框的类别一律不进清单
    labels = sorted({b.label for b in ctx.boxes})
    inventory = [(l, len(_same_label(ctx, l))) for l in labels]
    if not inventory:
        return None

    # 第二轮的目标：该类恰好一个实例，且本图还没用过
    picks = [l for l, n in inventory
             if n == 1 and _same_label(ctx, l)[0].index not in ctx.used]
    if not picks:
        return None
    label = ctx.rng.choice(picks)
    box = _same_label(ctx, label)[0]

    desc = _describe(ctx, box)
    if ctx.require_desc and not desc:
        return None

    listing = "、".join(f"{n}{ctx.mw(l)}{l}" for l, n in inventory)
    pairs = [
        (prompts.render_choice("inv_ask_what", ctx.rng),
         prompts.render_choice("inv_answer_what", ctx.rng, listing=listing)),
        (prompts.render_choice("inv_ask_box", ctx.rng, mw=ctx.mw(label), label=label),
         _j(_box_of(ctx, box), ctx.json_fence)),
    ]
    if desc:
        pairs.append((_describe_question(ctx, box), desc))
    return {"conversations": _turns(*pairs), "focus": [box.index], "label": label,
            "inventory": [f"{l}x{n}" for l, n in inventory]}


def _ground_describe(ctx: Ctx, kind: str):
    """指代 -> 框 -> 【某一种】描述。七个 ground_* 任务共用这一个实现。

    描述的多样性来自【答案里装的是什么】，不是问句换说法：
    appearance 只说物体本身、position 只说方位、relation 只说邻接关系……
    每种的答案信息结构完全不同。子类型由代码在预取阶段按配比指派给每个目标，
    模型做不到就返回空，这里取不到就跳过 —— 换下一张图的下一个目标。
    """
    cand = [b for b in ctx.unused(ctx.boxes)
            if (ctx.vlm.get(b.index) or {}).get("describe_kind") == kind
            and ctx.kind_fits(kind, b)
            and _attr_identifies(ctx, b, (ctx.vlm.get(b.index) or {}).get("attribute", ""))]
    if not cand:
        return None
    b = ctx.rng.choice(cand)
    info = ctx.vlm[b.index]
    attr = _clean_attr(info["attribute"])

    desc, desc_q = info.get("description", ""), info.get("describe_q", "")
    if not desc or not desc_q:
        return None            # 模型拒绝了这个子类型

    questions = [q.replace("的的", "的") for q in (info.get("questions") or []) if q]
    questions = [q for q in questions if _box_question_ok(ctx, q)]
    if questions:
        question, source = ctx.rng.choice(questions), "vlm"
    else:
        question = prompts.render_choice("ground_attribute", ctx.rng,
                                         attribute=attr, mw=ctx.mw(b.label),
                                         label=b.label)
        source = "template"

    return {"conversations": _turns(
                (_ask(ctx, question, box_answer=True), _j(_box_of(ctx, b), ctx.json_fence)),
                (desc_q, desc)),
            "focus": [b.index], "label": b.label, "attribute": attr,
            "describe_kind": kind, "question_source": source}


def _make_ground_task(kind: str):
    fn = lambda ctx, _k=kind: _ground_describe(ctx, _k)      # noqa: E731
    fn.__name__ = f"ground_{kind}"
    fn.__doc__ = f"指代 -> 框 -> {kind} 类描述。见 prompts/describe/{kind}.txt"
    return fn


# --------------------------------------------------------------- 其余七种
def detect_class(ctx: Ctx):
    """定位图中所有的人员。—— 答案必须完整，所以只在该类【全部框都合格】时才出。

    否则漏掉被质量过滤的框，等于在教模型漏检。
    """
    # 该类的框只要有一个已被别的样本用过，就不再出「定位所有该类」——
    # 这个任务的答案是该类的全部框，同一张图同一个类别只可能有一种答案，
    # 再出一条只是换了问法，属于近重复数据（实测同一答案出现过 6 次）。
    cand = [l for l in {b.label for b in ctx.boxes}
            if len(_same_label(ctx, l)) >= 2
            and ctx.all_kept(l)                       # 有一个被过滤掉，答案就不完整
            and all(b.index not in ctx.used for b in _same_label(ctx, l))]
    if not cand:
        return None
    label = ctx.rng.choice(sorted(cand))
    boxes = _same_label(ctx, label)
    return {"conversations": _turns(
                (_ask(ctx, prompts.render_choice("detect_class", ctx.rng, label=label),
                      box_answer=True),
                 _j([_box_of(ctx, b) for b in boxes], ctx.json_fence))),
            "focus": [b.index for b in boxes], "label": label, "n_boxes": len(boxes)}


def detect_describe(ctx: Ctx):
    """穷举定位 -> 坐标回指 -> 区域描述。

        用户 │ 框出图中所有的卡车。
        模型 │ [{"bbox_2d": [..]}, {"bbox_2d": [..]}, {"bbox_2d": [..]}]
        用户 │ 描述一下 {"bbox_2d": [..]} 这辆卡车。
        模型 │ 深红色车身，支着一顶白色遮阳篷……

    【第二轮用坐标回指，不用文字】第一轮的答案是同类的全部框，用文字说
    「那辆卡车」指不明是哪一辆 —— 这正是 ground_unique 卡死的地方
    （实测 997 次尝试零成功）。坐标天然无歧义，所以这个任务【不要求】
    类别在图中唯一，是部件级标注上唯一能成立的「多实例 + 描述」形态。

    第一轮的约束和 detect_class 完全一致（该类全部框合格、都没被用过），
    因为它就是 detect_class 的超集：同样的第一轮，多一轮描述。
    合并数据时同一个 (图, 类别) 保留本任务、丢弃 detect_class 那条，
    见 scripts/merge_by_group.py。
    """
    # 条件与 detect_class 一致 —— 少一条都会让第一轮的答案不完整或重复
    cand = [l for l in {b.label for b in ctx.boxes}
            if len(_same_label(ctx, l)) >= 2
            and ctx.all_kept(l)                       # 有一个被过滤掉，答案就不完整
            and all(b.index not in ctx.used for b in _same_label(ctx, l))]
    if not cand:
        return None

    # 只保留「至少有一个框拿得到描述」的类别 —— 第二轮没描述就退化成 detect_class，
    # 那还不如让 detect_class 自己出，不必多这一个任务。
    usable = [(l, [b for b in _same_label(ctx, l) if _describe(ctx, b)])
              for l in sorted(cand)]
    usable = [(l, bs) for l, bs in usable if bs]
    if not usable:
        return None

    label, describable = ctx.rng.choice(usable)
    boxes = _same_label(ctx, label)
    target = ctx.rng.choice(describable)
    desc = _describe(ctx, target)

    return {"conversations": _turns(
                (_ask(ctx, prompts.render_choice("detect_class", ctx.rng, label=label),
                      box_answer=True),
                 _j([_box_of(ctx, b) for b in boxes], ctx.json_fence)),
                (prompts.render_choice("detect_describe_pick", ctx.rng,
                                       bbox=_j(_box_of(ctx, target), False),
                                       mw=ctx.mw(label), label=label),
                 desc)),
            "focus": [b.index for b in boxes], "label": label,
            "n_boxes": len(boxes), "described_box": target.index}


def attribute_qa(ctx: Ctx):
    """[框] 这个区域内物体是什么颜色 / 有什么特征？—— Osprey-724K 里属性是
    最大一类（207K，29%）。用坐标指向目标，不存在把答案写进问题的可能。

    两个维度分开问，不能混：
      color      VLM 返回的主体颜色，配颜色问句
      attribute  VLM 返回的「用于指认的最显眼特征」，可能是颜色，也可能是
                 朝向、姿态、载货状态 —— 拿颜色问句去问它，会出现
                 「问什么颜色，答车头朝左」。

    两者都来自挑对象那次调用，主线已经付过钱，这里复用不额外花费。
    """
    dims = []
    for b in ctx.unused(ctx.boxes):
        info = ctx.vlm.get(b.index) or {}
        if info.get("color"):
            dims.append((b, "color", info["color"]))
        attr = _clean_attr(info.get("attribute", ""))
        # 特征和颜色一模一样时问「有什么特征」答「银灰色」，等于重复了颜色那条
        if attr and attr != info.get("color"):
            dims.append((b, "feature", attr))
    if not dims:
        return None
    b, kind, value = ctx.rng.choice(dims)
    pool = "attribute_qa_color" if kind == "color" else "attribute_qa_feature"
    answer = value if ctx.short_answer else (
        f"是{value}的。" if kind == "color" else f"{value}。")
    return {"conversations": _turns(
                (_ask(ctx, prompts.render_choice(pool, ctx.rng,
                                                 bbox=_j(ctx.bbox2d(b)))), answer)),
            "focus": [b.index], "label": b.label, "attribute": value,
            "attribute_kind": kind}


def spatial_relation(ctx: Ctx):
    """图中的银灰色卡车在人员的左边还是右边？—— 纯坐标计算（Ferret 四类之一）。

    只取关系明确的一对：一个轴拉开、另一个轴接近，否则「左边」这种回答不成立。

    三条讲究：

    1. 问法必须跟关系轴匹配。中文里「侧」指左右 —— 「在公交车的哪一侧？」
       答「在公交车的下方」是问非所答。左右一个问法池、上下一个问法池。

    2. 参照物 B 必须是该类在【原始标注】里的唯一实例，否则指不明确。

    3. 主体 A 的消歧优先用【外观属性】（「银灰色的那辆面包车」），
       实在没有才回落方位修饰。原先一律用「中部左侧那辆面包车」，一句话里
       塞两套方位体系：「中部左侧那辆面包车在公交车的下方」，读的人要同时
       转换两个坐标系。而且方位修饰和关系轴撞上时会直接打架
       （「左边那辆车在卡车的左侧」）—— 回落时避开同轴。

    属性来自挑对象那次调用，主线已经付过钱了，这里复用不额外花费。
    """
    singles = {l for l in {b.label for b in ctx.boxes}
               if len(_same_label(ctx, l)) == 1 and ctx.all_kept(l)}
    if not singles:
        return None

    pool = []
    for a in ctx.boxes:
        for b in ctx.boxes:
            if a.index == b.index or a.label == b.label or b.label not in singles:
                continue
            dx, dy = a.cx - b.cx, a.cy - b.cy
            if abs(dx) > 0.12 and abs(dy) < 0.12:
                pool.append((a, b, "右侧" if dx > 0 else "左侧", "lr"))
            elif abs(dy) > 0.12 and abs(dx) < 0.12:
                pool.append((a, b, "下方" if dy > 0 else "上方", "ud"))
    if not pool:
        return None

    ctx.rng.shuffle(pool)
    for a, b, rel, axis in pool:
        subject = _spatial_subject(ctx, a, axis)
        if subject:
            break
    else:
        return None

    question = prompts.render_choice(f"spatial_ask_{axis}", ctx.rng,
                                     a=subject, b=b.label)
    answer = (rel if ctx.short_answer else
              prompts.render_choice("spatial_answer", ctx.rng,
                                    a=subject, b=b.label, rel=rel))
    return {"conversations": _turns((_ask(ctx, question), answer)),
            "focus": [a.index, b.index], "label": a.label, "relation": rel,
            "relation_axis": axis}


# 方位修饰与关系轴同轴时会打架：「左边那辆车在卡车的左侧」。
# 回落时只用另一根轴上的词。
_ZONE_WORDS = {"lr": ("上方", "下方"), "ud": ("左侧", "右侧")}


def _spatial_subject(ctx: Ctx, a, axis: str) -> Optional[str]:
    """给主体 A 一个能指明白的说法。全图同类唯一就直接用类别名。"""
    mw = ctx.mw(a.label)
    if len(_same_label(ctx, a.label)) == 1:
        return a.label
    attr = _clean_attr((ctx.vlm.get(a.index) or {}).get("attribute", ""))
    if attr:
        return f"{attr}的那{mw}{a.label}"
    # 没有属性：用【另一根轴】上的方位词消歧，且必须在该轴上真的分得开
    lo, hi = _ZONE_WORDS[axis]
    v = a.cy if axis == "lr" else a.cx
    same = [o for o in _same_label(ctx, a.label) if o.index != a.index]
    others = [(o.cy if axis == "lr" else o.cx) for o in same]
    if others and all(abs(v - o) > 0.2 for o in others):
        return f"{lo if v < 0.5 else hi}那{mw}{a.label}"
    return None


def exist_negative(ctx: Ctx):
    """图中有没有直升机？—— 不可省。没有这类样本，模型会学到「被问就一定有」，
    推理时凭空编框。Ferret 有 95K hard negative（8.6%），Osprey 有 64K（8.8%）。

    一半问存在的类别（答有），一半问不存在的（答没有），避免模型学成「一律答没有」。

    问不存在的类别时【优先挑易混类别】。从 347 类里随机抽，抽到的多半是跟画面
    毫不相干的东西（航拍街景问「有没有N95防护口罩」），模型答「没有」不需要
    真的看图，学不到东西。挑「图里有卡车，问有没有货车」这种，才逼它去看。
    """
    present = {b.label for b in ctx.boxes}
    if ctx.rng.random() < 0.5 and present:
        label = ctx.rng.choice(sorted(present))
        if ctx.short_answer:
            a = "有"
        elif ctx.all_kept(label):
            a = prompts.render_choice("exist_yes_answer", ctx.rng,
                                      n=len(_same_label(ctx, label)),
                                      mw=ctx.mw(label), label=label)
        else:
            # 该类有实例被质量过滤掉了，报出来的数一定小于图里实际的个数。
            a = prompts.render_choice("exist_yes_vague", ctx.rng, label=label)
        polarity, hard = "positive", False
    else:
        # 上下位词要从【整个】拒答池里排掉，不只是难负样本那一路：
        # 图里有遮阳三轮车，问「有没有三轮车」答「没有」是错的，
        # 随机兜底那一路照样会抽到它。
        banned = set(present) | ctx.hypernyms_of(present)
        hard_pool = [l for l in ctx.confusable_with(present) if l not in banned]
        pool = hard_pool or [l for l in ctx.all_labels if l not in banned]
        if not pool:
            return None
        label = ctx.rng.choice(sorted(pool))
        hard = bool(hard_pool)
        a = ("没有" if ctx.short_answer else
             prompts.render_choice("exist_no_answer", ctx.rng, label=label))
        polarity = "negative"
    return {"conversations": _turns(
                (_ask(ctx, prompts.render_choice("exist_ask", ctx.rng, label=label)), a)),
            "focus": [], "label": label, "polarity": polarity,
            "hard_negative": hard}


def region_identify(ctx: Ctx):
    """图中 [框] 这个位置的是什么？—— REG 方向，用坐标指向目标。

    没有文字指代，因此不存在任何「把答案写进问题」的可能。
    """
    cand = ctx.unused(ctx.boxes)
    if not cand:
        return None
    b = ctx.rng.choice(cand)
    a = b.label if ctx.short_answer else prompts.render_choice("region_identify_answer", ctx.rng, label=b.label)
    return {"conversations": _turns(
                (_ask(ctx, prompts.render_choice("region_identify", ctx.rng,
                                                 bbox=_j(ctx.bbox2d(b)))), a)),
            "focus": [b.index], "label": b.label}


# 七个描述子类型各是一个任务 —— 配比可控、报告里看得到每种多少条、
# 可以按 task_type 直接筛掉不要的。名字统一 ground_<子类型>，
# 对应 prompts/describe/<子类型>.txt。
DESCRIBE_KINDS = ("appearance", "state", "part", "position",
                  "relation", "contrast", "full")

TASKS: Dict[str, Callable[[Ctx], Optional[Dict[str, Any]]]] = {
    "ground_unique": ground_unique,
    "inventory_locate": inventory_locate,
    **{f"ground_{k}": _make_ground_task(k) for k in DESCRIBE_KINDS},
    "detect_class": detect_class,
    "detect_describe": detect_describe,
    "attribute_qa": attribute_qa,
    "spatial_relation": spatial_relation,
    "exist_negative": exist_negative,
    "region_identify": region_identify,
}

# 主线 = 以【区域描述】收尾的那几个。detect_describe 也在内：它同样是
# 「锁定目标 -> 坐标 -> 描述」，只是上游换成了单类穷举而不是文字指代。
MAIN_LINE = ("ground_unique", "inventory_locate", "detect_describe",
             *(f"ground_{k}" for k in DESCRIBE_KINDS))

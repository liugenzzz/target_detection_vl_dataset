"""目标难度分级与配额采样。

需求：数据集以清晰易识别的目标为主，困难目标控制在 10% 左右。

为什么不用硬阈值一刀切：
  - 全部剔除困难目标 -> 模型在真实场景遇到小目标/密集目标时直接崩，
    因为训练时从没见过。
  - 不加控制 -> 困难目标会主导数据分布（实测 COCO128 里 24% 的框短边不足
    16px、56% 属于同类密集），模型学不好基础定位。
  所以按难度分档，再按配额采样，让困难目标占既定比例。

关键设计：尺寸难度用【面积占比】而不是绝对像素。
  业务数据分辨率跨度极大（128x128 ~ 2048x1440）。同一个目标（占图宽 6.6%、
  高 2.8%）在 128px 原图上只有 3.6px、在 2048px 原图上有 40.8px —— 绝对像素
  阈值会对同一个目标给出相反判定。而 Qwen3-VL 会把图缩放到自己的输入分辨率，
  模型实际看到多少像素只由面积占比决定，与原图尺寸无关（上例中恒为 44px）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

# Qwen3-VL 输入侧的等效边长基准。用于把「面积占比」换算成
# 「模型实际看到的像素」，让阈值有直观的物理含义。
QWEN_EQUIV_SIZE = 1024

EASY = "easy"
MEDIUM = "medium"
HARD = "hard"
REJECT = "reject"

_ORDER = {EASY: 0, MEDIUM: 1, HARD: 2, REJECT: 3}

# 尺寸分档，按模型实际看到的等效像素
SIZE_EASY_PX = 64      # 面积占比 >= 0.39%
SIZE_MEDIUM_PX = 32    # 面积占比 >= 0.098%
SIZE_REJECT_PX = 16    # 面积占比 <  0.024% -> 剔除

# 同图同类目标数分档
DENSE_MEDIUM = 3       # 2~3 个同类
DENSE_HARD = 8         # 4~8 个同类；超过则无法可靠指代 -> 剔除

# 单个框占整图面积上限。业务标注里存在占满全图的框，
# 拿来做定位样本没有意义（答案是整张图）。
MAX_AREA_RATIO = 0.5

# 困难目标配额
HARD_QUOTA = 0.10


def equivalent_px(area_ratio: float) -> float:
    """面积占比 -> 模型实际看到的等效边长像素。对原图缩放不变。"""
    return (max(area_ratio, 0.0) ** 0.5) * QWEN_EQUIV_SIZE


def size_grade(area_ratio: float) -> str:
    if area_ratio > MAX_AREA_RATIO:
        return REJECT                      # 全图框
    px = equivalent_px(area_ratio)
    if px >= SIZE_EASY_PX:
        return EASY
    if px >= SIZE_MEDIUM_PX:
        return MEDIUM
    if px >= SIZE_REJECT_PX:
        return HARD
    return REJECT


def density_grade(same_label_count: int) -> str:
    if same_label_count <= 1:
        return EASY
    if same_label_count <= DENSE_MEDIUM:
        return MEDIUM
    if same_label_count <= DENSE_HARD:
        return HARD
    return REJECT


def referring_grade(unique_in_zone: bool) -> str:
    """3x3 空间分区内同类唯一 -> 模板空间指代可用（简单）；
    否则需要调 VLM 生成视觉指代（困难）。"""
    return EASY if unique_in_zone else HARD


def combine(*grades: str) -> str:
    """总难度取各维度里最难的一项。"""
    return max(grades, key=lambda g: _ORDER[g])


@dataclass
class GradedBox:
    index: int
    label: str
    grade: str
    area_ratio: float
    equiv_px: float
    same_label_count: int
    unique_in_zone: bool
    reasons: Dict[str, str]


def zone_of(cx: float, cy: float) -> str:
    h = "左" if cx < 1 / 3 else "右" if cx > 2 / 3 else "中"
    v = "上" if cy < 1 / 3 else "下" if cy > 2 / 3 else "中"
    return v + h


def grade_image(boxes: Sequence[dict]) -> List[GradedBox]:
    """给一张图里的所有框定级。

    boxes 每项需含 label 和归一化的 cx/cy/w/h（YOLO 原生格式）。
    """
    labels = [b["label"] for b in boxes]
    zones = [(b["label"], zone_of(b["cx"], b["cy"])) for b in boxes]

    out: List[GradedBox] = []
    for i, b in enumerate(boxes):
        area = b["w"] * b["h"]
        same = labels.count(b["label"])
        uniq = zones.count(zones[i]) == 1

        g_size = size_grade(area)
        g_dense = density_grade(same)
        g_ref = referring_grade(uniq)

        out.append(GradedBox(
            index=i,
            label=b["label"],
            grade=combine(g_size, g_dense, g_ref),
            area_ratio=area,
            equiv_px=equivalent_px(area),
            same_label_count=same,
            unique_in_zone=uniq,
            reasons={"size": g_size, "density": g_dense, "referring": g_ref},
        ))
    return out


def pick_candidates(
    graded: Sequence[GradedBox],
    cap: int,
    seed: int = 20260826,
) -> List[GradedBox]:
    """第一阶段（逐图）：挑出这张图最多 cap 个候选框，简单档优先。

    这里只做「每图取多少」的限制，不做难度配额 —— 配额必须放到全局做，
    见 balance_hard_quota()。逐图配额是行不通的：数据里存在大量整张图
    全是困难目标的图，逐图配额时只能全取困难的，最后困难占比会失控
    （实测 COCO128 上逐图配额得到 40.7% 困难目标，而目标是 10%）。
    """
    import random

    rng = random.Random(seed)
    pool = [g for g in graded if g.grade != REJECT]
    if not pool:
        return []

    easy = [g for g in pool if g.grade in (EASY, MEDIUM)]
    hard = [g for g in pool if g.grade == HARD]
    rng.shuffle(easy)
    rng.shuffle(hard)

    chosen = (easy + hard)[:cap]          # 简单档优先占满 cap
    return sorted(chosen, key=lambda g: g.index)


def balance_hard_quota(
    candidates: Sequence,
    grade_of,
    hard_quota: float = HARD_QUOTA,
    seed: int = 20260826,
) -> List:
    """第二阶段（全局）：把困难目标下采样到 hard_quota 占比。

    candidates 是全部图汇总后的候选列表，grade_of(item) 取出该项的难度档。
    简单档全保留，困难档按比例截断：
        hard / (easy + hard) = quota  =>  hard = easy * quota / (1 - quota)

    返回打乱顺序后的最终列表（固定种子，可复现）。
    """
    import random

    rng = random.Random(seed)
    easy = [c for c in candidates if grade_of(c) in (EASY, MEDIUM)]
    hard = [c for c in candidates if grade_of(c) == HARD]

    if hard_quota <= 0:
        keep_hard = 0
    elif hard_quota >= 1:
        keep_hard = len(hard)
    else:
        keep_hard = min(len(hard), int(round(len(easy) * hard_quota / (1 - hard_quota))))

    rng.shuffle(hard)
    result = easy + hard[:keep_hard]
    rng.shuffle(result)
    return result

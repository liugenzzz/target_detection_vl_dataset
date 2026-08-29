"""目标难度分级与困难目标配额采样。

需求：数据集以清晰易识别的目标为主，困难目标控制在配置的比例（默认 10%）。

为什么不用硬阈值一刀切：
  - 全部剔除困难目标 -> 模型在真实场景遇到小目标/密集目标时直接崩。
  - 不加控制 -> 困难目标会主导分布（实测 COCO128 里 24% 的框短边不足 16px、
    56% 属于同类密集；VisDrone 上短边中位数只有 18px）。

尺寸维度用【面积占比】而不是绝对像素：业务数据分辨率跨度 128x128~2048x1440，
同一个目标（占图宽 6.6%、高 2.8%）在 128px 原图上仅 3.6px、在 2048px 原图上
有 40.8px —— 绝对像素阈值会对同一目标给出相反判定。而模型会把图缩放到自身
输入分辨率，实际看到多少像素只由面积占比决定（上例中恒为 44px）。

配额必须在全局做，不能逐图做：数据里存在大量整张图全是困难目标的图，
逐图配额时只能全取困难的。实测 COCO128 上逐图配额得到 40.7% 困难目标，
改成两阶段后精确命中 10%。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from .coords import zone_of

EASY, MEDIUM, HARD, REJECT = "easy", "medium", "hard", "reject"
_ORDER = {EASY: 0, MEDIUM: 1, HARD: 2, REJECT: 3}


def grade_rank(grade: str) -> int:
    """难度档位的序，easy < medium < hard。用于「取最难的那个」。"""
    return _ORDER.get(grade, 0)


def grade_at_most(grade: str, limit: str) -> bool:
    """grade 是否不难于 limit。limit 为空表示不限。

    描述子类型用它来卡适用档位：`part`（聚焦部位）和 `contrast`（同类对比）
    在困难目标上必然写不出来 —— 看不清部件、也分辨不出和同类的差别，
    模型硬写就是编。
    """
    if not limit:
        return True
    return _ORDER.get(grade, 99) <= _ORDER.get(limit, 99)


# 报表用的尺寸分档，和难度分档是【两根独立的轴】：
# 难度还掺了密集度和指代唯一性，尺寸只看框有多大。评估报告要按尺寸拆开看
# （小目标掉点是不是特别厉害），所以单独给一套阈值。
# 阈值沿用 COCO 的 32 / 96 边长口径，但作用在 equiv_px 上 —— 那是「面积占比
# 还原到 1024 基准的等效边长」，对原图分辨率不变，128x128 和 2048x1440
# 两张图上的同一个占比会落进同一档。直接用原图绝对像素做不到这点。
SMALL, MEDIUM_SIZE, LARGE = "small", "medium", "large"


def size_bucket(equiv_px: float, small_px: float = 32.0, large_px: float = 96.0) -> str:
    if equiv_px < small_px:
        return SMALL
    if equiv_px < large_px:
        return MEDIUM_SIZE
    return LARGE


@dataclass
class Grade:
    box_index: int
    label: str
    grade: str
    area_ratio: float
    equiv_px: float
    same_label_count: int
    unique_in_zone: bool
    zone: str
    reasons: Dict[str, str]


class Grader:
    """按配置分级。所有阈值来自 config，服务器上改 yaml 即可调整。"""

    def __init__(self, cfg):
        d = cfg.get_path("difficulty", {}) or {}
        q = cfg.get_path("quality", {}) or {}
        self.equiv_size = float(d.get("qwen_equiv_size", 1024))
        self.size_easy = float(d.get("size_easy_px", 64))
        self.size_medium = float(d.get("size_medium_px", 32))
        self.size_reject = float(d.get("size_reject_px", 16))
        self.dense_medium = int(d.get("dense_medium", 3))
        self.dense_hard = int(d.get("dense_hard", 8))
        self.hard_quota = float(d.get("hard_quota", 0.10))
        self.bucket_small = float(d.get("size_bucket_small_px", 32))
        self.bucket_large = float(d.get("size_bucket_large_px", 96))
        self.min_area = float(q.get("min_area_ratio", 0.001))
        self.max_area = float(q.get("max_area_ratio", 0.5))
        self.min_short_px = float(q.get("min_short_side_px", 16))

    def bucket_of(self, grade: "Grade") -> str:
        """这个框的尺寸档，供报表按尺寸拆分。"""
        return size_bucket(grade.equiv_px, self.bucket_small, self.bucket_large)

    def equivalent_px(self, area_ratio: float) -> float:
        """面积占比 -> 模型实际看到的等效边长像素。对原图缩放不变。"""
        return (max(area_ratio, 0.0) ** 0.5) * self.equiv_size

    def _size_grade(self, area_ratio: float, short_px: float) -> str:
        if area_ratio > self.max_area:
            return REJECT                       # 全图框，做定位样本无意义
        if area_ratio < self.min_area:
            return REJECT
        if short_px < self.min_short_px:
            return REJECT                       # 原图上本身就糊
        px = self.equivalent_px(area_ratio)
        if px >= self.size_easy:
            return EASY
        if px >= self.size_medium:
            return MEDIUM
        if px >= self.size_reject:
            return HARD
        return REJECT

    def _density_grade(self, same_label_count: int) -> str:
        if same_label_count <= 1:
            return EASY
        if same_label_count <= self.dense_medium:
            return MEDIUM
        if same_label_count <= self.dense_hard:
            return HARD
        return REJECT                           # 太密集，无法可靠指代

    @staticmethod
    def _referring_grade(unique_in_zone: bool) -> str:
        """3x3 分区内同类唯一 -> 模板空间指代可用（简单）；
        否则需调 VLM 生成视觉指代（困难）。"""
        return EASY if unique_in_zone else HARD

    def grade_image(self, boxes: Sequence, img_w: int, img_h: int) -> List[Grade]:
        """给一张图里的所有框定级。boxes 为 core.yolo.Box 列表。"""
        labels = [b.label for b in boxes]
        zones = [(b.label, zone_of(b.cx, b.cy)) for b in boxes]

        out: List[Grade] = []
        for i, b in enumerate(boxes):
            area = b.area_ratio
            same = labels.count(b.label)
            uniq = zones.count(zones[i]) == 1
            g_size = self._size_grade(area, b.short_side_px(img_w, img_h))
            g_dense = self._density_grade(same)
            g_ref = self._referring_grade(uniq)
            out.append(Grade(
                box_index=b.index, label=b.label,
                grade=max((g_size, g_dense, g_ref), key=lambda g: _ORDER[g]),
                area_ratio=area, equiv_px=self.equivalent_px(area),
                same_label_count=same, unique_in_zone=uniq, zone=zones[i][1],
                reasons={"size": g_size, "density": g_dense, "referring": g_ref},
            ))
        return out


def pick_candidates(graded: Sequence[Grade], cap: int, seed: int = 0) -> List[Grade]:
    """第一阶段（逐图）：挑出最多 cap 个候选，简单档优先。这里不做配额。"""
    pool = [g for g in graded if g.grade != REJECT]
    if not pool:
        return []
    easy = [g for g in pool if g.grade in (EASY, MEDIUM)]
    hard = [g for g in pool if g.grade == HARD]
    rng = random.Random(seed)
    rng.shuffle(easy)
    rng.shuffle(hard)
    return sorted((easy + hard)[:cap], key=lambda g: g.box_index)


def balance_hard_quota(candidates: Sequence, grade_of: Callable,
                       hard_quota: float = 0.10, seed: int = 0) -> List:
    """第二阶段（全局）：把困难目标下采样到 hard_quota 占比。

    简单档全保留，困难档按比例截断：
        hard / (easy + hard) = quota  =>  hard = easy * quota / (1 - quota)
    """
    rng = random.Random(seed)
    easy = [c for c in candidates if grade_of(c) in (EASY, MEDIUM)]
    hard = [c for c in candidates if grade_of(c) == HARD]

    if hard_quota <= 0:
        keep = 0
    elif hard_quota >= 1:
        keep = len(hard)
    else:
        keep = min(len(hard), int(round(len(easy) * hard_quota / (1 - hard_quota))))

    rng.shuffle(hard)
    result = easy + hard[:keep]
    rng.shuffle(result)
    return result

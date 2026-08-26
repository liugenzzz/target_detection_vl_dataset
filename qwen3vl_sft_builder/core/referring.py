"""指代短语生成。

分两档：
  1. 空间指代（模板，零成本）—— 目标在 3x3 分区内同类唯一时可用。
  2. 视觉指代（VLM 生成）—— 同类多目标时，靠外观特征区分。

明确不做「第 N 个」序号指代：序号来自代码里的排序规则，图像里没有任何
视觉线索可以推断，模型永远学不会，训练这类样本等于教模型瞎猜。
"""

from __future__ import annotations

from typing import Optional

from .coords import zone_of


def spatial_phrase(cx: float, cy: float) -> str:
    """『中部中间』『上方左侧』这类空间短语。"""
    zone = zone_of(cx, cy)
    vert = {"上": "上方", "中": "中部", "下": "下方"}[zone[0]]
    horiz = {"左": "左侧", "中": "中间", "右": "右侧"}[zone[1]]
    return f"{vert}{horiz}"


def template_referring(cx: float, cy: float, unique_in_zone: bool,
                       label: Optional[str] = None) -> str:
    """模板指代。分区内唯一时用纯空间指代；否则带上类别名缩小范围
    （仍可能有歧义，这种情况应该交给 VLM 生成视觉指代）。"""
    base = spatial_phrase(cx, cy)
    if unique_in_zone:
        return f"{base}那个目标"
    return f"{base}那个{label}" if label else f"{base}那个目标"


def template_description(label: str, cx: float, cy: float,
                         equiv_px: float, small_px: float = 32.0) -> str:
    """模板描述，作为 VLM 不可用或失败时的兜底。

    只陈述由标注本身可确定的事实（类别、位置、相对大小），
    绝不编造外观属性 —— 编出来的描述比没有描述更有害。
    """
    hint = "，目标尺寸较小" if equiv_px < small_px else ""
    return f"{spatial_phrase(cx, cy)}是一个{label}{hint}。"

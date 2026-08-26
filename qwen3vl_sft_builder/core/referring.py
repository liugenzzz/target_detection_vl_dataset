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


def leaks_label(referring: str, label: str) -> bool:
    """指代短语里有没有把类别名说出来。

    第一轮问的是「图中{referring}是什么？」，答案是类别名。如果指代短语里
    已经出现了类别名，等于把答案写进了问题里，模型学不到识别能力 ——
    只学会把「三轮车」翻译成 tricycle。这类样本是废的，必须拦下来。

    提示词里已经明令禁止，但提示词管不住模型，代码这一层必须兜底。

    三档判据，都只对中文类别名生效（英文类别名和中文指代没法这样比对）：

      1. 类别名整体出现          「靠近树木的那辆三轮车」/ 三轮车
      2. 两字以上的尾缀出现       「穿红色上衣的那个人员」/ 军事人员
      3. 单字尾缀出现在【结尾】   「画面中部那艘船」/ 其它辅助船

    第 3 条必须限定在结尾，否则「车」「船」这类高频字会大量误判 ——
    「停在斜坡上、车头朝左的那个目标」里的「车头」并没有泄漏「卡车」。
    """
    ref = (referring or "").strip()
    lab = (label or "").strip()
    if not ref or not lab:
        return False
    if lab in ref:
        return True
    if not _is_cjk(lab):
        return False

    for cut in range(1, len(lab)):
        tail = lab[cut:]
        if len(tail) >= 2:
            if tail in ref:
                return True
        elif ref.endswith(tail):        # 单字尾缀只认结尾，避免高频字误判
            return True
    return False


def _is_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


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

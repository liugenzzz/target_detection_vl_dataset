"""描述语句的成色判定与空间措辞。

原先这里还有一整套【指代泄漏】检查（指代短语里不能出现类别名或类别暗示词），
服务于最早那版设计 —— 第一轮拿文字指代去问类别（「靠近树木的那个是什么？」），
指代里带上类别名就等于把答案写进了问题。

多任务改造之后没有任何任务是这个形状了：主线问的是【坐标】，
问句里出现类别名是正确的（RefCOCO/Qwen grounding 数据都这么写，
答案是框不是类别）；反过来问类别的 region_identify 用坐标指向目标，
根本没有文字指代。所以那套检查连同它服务的旧路径一起删掉了。

明确不做「第 N 个」序号指代：序号来自代码里的排序规则，图像里没有任何
视觉线索可以推断，模型永远学不会，训练这类样本等于教模型瞎猜。
"""

from __future__ import annotations

from .coords import zone_of

# 套话短语。这些词放到任何一张图、任何一个目标上都成立，
# 通篇只有它们就等于什么也没说。
_VACUOUS = (
    "一处目标", "一个目标", "轮廓清晰", "位于道路旁", "在道路旁",
    "画面中一处", "位于画面中", "清晰可见", "较为明显",
)


def is_vacuous_description(desc: str, min_len: int = 18) -> bool:
    """描述是不是空话。

    实测模型会写出「画面中一处目标，轮廓清晰，位于道路旁。」——
    三个分句全是套话，放到任何一张图任何一个目标上都成立，照着找不到东西。

    两条判据：太短，或者通篇只有套话短语、没有任何具体参照物。
    """
    d = (desc or "").strip()
    if not d:
        return True
    if len(d) < min_len:
        return True
    # 去掉套话短语后还剩多少实质内容
    rest = d
    for w in _VACUOUS:
        rest = rest.replace(w, "")
    rest = rest.strip("，。、,. 　")
    return len(rest) < min_len // 2



def _is_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def spatial_phrase(cx: float, cy: float) -> str:
    """『中部中间』『上方左侧』这类空间短语。"""
    zone = zone_of(cx, cy)
    vert = {"上": "上方", "中": "中部", "下": "下方"}[zone[0]]
    horiz = {"左": "左侧", "中": "中间", "右": "右侧"}[zone[1]]
    return f"{vert}{horiz}"




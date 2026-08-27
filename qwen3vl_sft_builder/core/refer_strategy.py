"""指代策略决策：先判断该用哪种指代方式，再生成骨架短语。

这一步比句式库重要得多。人指认目标是有优先级的，不是见谁都套一句方位：

    同类只有一个        直接说类别，绝不会说「左上角那个」
    同类两三个          用极值：「最左边那个」「最靠前那个」
    同类很多            极值 + 粗方位：「下面那排最左边的」
    旁边有显著异类      用锚点：「卡车旁边那个」
    目标贴着画面边缘    边缘特例：「贴着左边缘那个」
    以上都不成立        这个目标没法用语言指认，丢弃

判据全部可以从 YOLO 标注直接算出来：同类计数、中心点排序、最近邻异类框距离、
到画面边缘的距离。不需要看图，因此这一步零成本。

刻意不做【序数】指代（「从左数第二辆」）。序数要求模型和我们建立同一套排序
规则，而图像里没有这个线索，模型学不会 —— 只有【极值】（最左/最右/最上/最下）
是视觉上可判断的。

生成的骨架**不含目标自身的类别名**，因此可以拿来问「这是什么」而不泄漏答案。
锚点里出现的是【别的】类别（「卡车旁边那个」），不构成泄漏。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

# 中心点离任一画面边缘小于这个比例，判为贴边
EDGE_MARGIN = 0.08
# 异类目标中心距小于这个比例，才算「旁边」，可以做锚点
ANCHOR_DIST = 0.15
# 同类超过这个数量，光靠极值不够，要再加粗方位
CROWD_THRESHOLD = 3


@dataclass
class Skeleton:
    strategy: str        # unique / extreme / extreme_zone / anchor / edge
    phrase: str          # 骨架短语，如「最左边那个」，不含目标类别名
    anchor_label: str = ""   # 锚点策略用到的那个异类的类别名

    @property
    def bare(self) -> str:
        """去掉末尾「那个」的骨架。句式模板自己要接名词时用这个，
        否则会拼出「卡车旁边那个那玩意儿」。"""
        return self.phrase[:-2] if self.phrase.endswith("那个") else self.phrase


def _coarse_zone(cx: float, cy: float, axis: str = "auto") -> str:
    """粗方位。人说「右边那个」远多于「右上角那个」，所以只给一个维度。

    axis 指定用哪个轴：极值词已经交代了一个轴时，粗方位要用【另一个】轴，
    否则会拼出「左边最左边那个」这种废话。
    """
    if axis == "x" or (axis == "auto" and abs(cx - 0.5) >= abs(cy - 0.5)):
        return "右边" if cx > 0.5 else "左边"
    return "下面" if cy > 0.5 else "上面"


def _extreme_word(box, pool: Sequence):
    """该目标在 pool 里是否处于某个轴的极端。

    返回 (说法, 该说法用掉的轴)，例如 ("最左边", "x")。轴要返回出去，
    是为了让粗方位改用另一个轴，避免拼出「左边最左边那个」。
    """
    if len(pool) < 2:
        return None
    if box is min(pool, key=lambda b: b.cx):
        return "最左边", "x"
    if box is max(pool, key=lambda b: b.cx):
        return "最右边", "x"
    if box is min(pool, key=lambda b: b.cy):
        return "最上面", "y"
    if box is max(pool, key=lambda b: b.cy):
        return "最下面", "y"
    return None


def _dist(a, b) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def _nearest_unique_other(box, boxes, label_counts) -> Optional[Any]:
    """最近的、可以当锚点的异类目标。两个条件缺一不可：

    1. 该异类在图中唯一 ——「卡车旁边那个」里的卡车必须只有一辆，
       否则锚点自己就是歧义的。
    2. 本目标是该锚点附近【唯一】的目标 —— 否则人员和三轮车都挨着那辆卡车时，
       两者都会拿到「卡车旁边那个」，指代撞车。
    """
    best, best_d = None, ANCHOR_DIST
    for o in boxes:
        if o.label == box.label or label_counts.get(o.label, 0) != 1:
            continue
        d = _dist(o, box)
        if d < best_d:
            best, best_d = o, d
    if best is None:
        return None
    others = [x for x in boxes
              if x is not box and x is not best and _dist(best, x) < ANCHOR_DIST]
    return None if others else best


def decide(box, boxes: Sequence, label_counts) -> Optional[Skeleton]:
    """给一个目标决定指代策略并生成骨架。无策略可用时返回 None（该目标丢弃）。"""
    n = label_counts.get(box.label, 0)

    if n == 1:
        # 同类唯一：人会直接说类别名。这种情况骨架必然含类别名，
        # 不能拿来问「这是什么」，交给 ground_unique 出定位样本。
        return Skeleton("unique", box.label)

    anchor = _nearest_unique_other(box, boxes, label_counts)
    if anchor is not None:
        return Skeleton("anchor", f"{anchor.label}旁边那个", anchor.label)

    # 极值必须在【全图】范围算，不能在同类内部算。
    # 骨架刻意不带类别名（这样才能问「这是什么」而不泄漏答案），
    # 于是「最左边那个」在人看来指的是全图最左的目标，而不是「最左的那辆三轮车」。
    # 按同类算会产出「最左边那个」却给出一个偏右的框 —— 实测 29% 的样本自相矛盾。
    found = _extreme_word(box, boxes)
    if found:
        return Skeleton("extreme", f"{found[0]}那个")

    # 全图极值只有四个名额，不够用。退一步：先按粗方位把画面切成两半，
    # 再在那一半里取极值 —— 「右边最下面那个」指的是画面右半边里最靠下的那个。
    for axis in ("x", "y"):
        zone = _coarse_zone(box.cx, box.cy, axis)
        if axis == "x":
            half = [b for b in boxes if (b.cx > 0.5) == (box.cx > 0.5)]
            other = "y"
        else:
            half = [b for b in boxes if (b.cy > 0.5) == (box.cy > 0.5)]
            other = "x"
        found = _extreme_word(box, half)
        if found and found[1] == other:
            return Skeleton("extreme_zone", f"{zone}{found[0]}那个")

    if min(box.cx, box.cy, 1 - box.cx, 1 - box.cy) < EDGE_MARGIN:
        side = {"左边": "左", "右边": "右", "上面": "上", "下面": "下"}[
            _coarse_zone(box.cx, box.cy)]
        return Skeleton("edge", f"贴着{side}边缘那个")

    return None


# 可以拿来问「这是什么」的策略：骨架里不含目标自身的类别名
ASKABLE = ("extreme", "extreme_zone", "anchor", "edge")


def decide_all(boxes: Sequence, label_counts) -> dict:
    """给整张图所有目标定策略，并保证骨架在全图唯一。

    单个目标看着没问题的骨架，放到全图可能撞车 —— 人员里的「最左边那个」和
    三轮车里的「最左边那个」是两句一模一样的话，指向却是两个目标。
    骨架不含类别名正是为了能问「这是什么」，代价就是跨类别更容易重名，
    所以必须在全图范围内查一遍，重名的全部丢弃（宁可少出样本，不留歧义）。

    返回 {box_index: Skeleton}，只含骨架唯一的目标。
    """
    got = {}
    for b in boxes:
        sk = decide(b, boxes, label_counts)
        if sk is not None:
            got[b.index] = sk

    by_index = {b.index: b for b in boxes}

    def collisions(d):
        seen = {}
        for i, sk in d.items():
            seen.setdefault(sk.phrase, []).append(i)
        return {p: idxs for p, idxs in seen.items() if len(idxs) > 1}

    # 撞车的先尝试补一个粗方位消歧：两个「最左边那个」分处上下时，
    # 「上面最左边那个」和「下面最左边那个」就分得开了。
    # 直接丢弃太浪费 —— 实测 extreme 这一支会因此只剩 43%。
    for phrase, idxs in collisions(got).items():
        for i in idxs:
            sk, b = got[i], by_index[i]
            if sk.strategy != "extreme":
                continue
            axis = "y" if ("左" in sk.phrase or "右" in sk.phrase) else "x"
            zone = _coarse_zone(b.cx, b.cy, axis)
            got[i] = Skeleton("extreme_zone", f"{zone}{sk.phrase}")

    # 补完还撞的，全部丢弃 —— 宁可少出样本，不留歧义
    for phrase, idxs in collisions(got).items():
        for i in idxs:
            got.pop(i, None)
    return got

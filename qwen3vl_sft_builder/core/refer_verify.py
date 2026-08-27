"""指代表达的反向验证。

这是 ReferItGame 那个「一个人写指代、另一个人照着点，点对了才收录」机制的
自动化版本：把生成好的指代表达连同原图喂回模型，让它输出框，和原框的 IoU
超过阈值才保留这条样本。

前面的策略决策、句式库、口语化改写都是启发式 —— 只有这一步是可证伪的质量闸，
能刷掉人看不懂或者指代歧义的表达。

务必先跑 scripts/calibrate_verifier.py 标定：验证器本身的定位能力有限，
IoU 不达标可能是指代不好，也可能是验证器不准。不标定就设阈值会大量误杀。
"""

from __future__ import annotations

import json
import re
from typing import Optional, Sequence


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两个 [x1,y1,x2,y2] 框的交并比。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_box(text: str) -> Optional[list]:
    """从验证器的回复里抠出 bbox_2d。模型常带 ```json 包裹或解释性前后缀。"""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            box = data.get("bbox_2d") if isinstance(data, dict) else None
            if isinstance(box, (list, tuple)) and len(box) == 4:
                return [float(v) for v in box]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # 退一步：直接找四个数字。模型有时只吐 [12,34,56,78]
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if len(nums) >= 4:
        return [float(v) for v in nums[:4]]
    return None


def normalize(box: Sequence[float], scale: int = 1000) -> list:
    """保证 x1<x2、y1<y2 并裁剪到 [0, scale]。"""
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    clip = lambda v: max(0.0, min(float(scale), v))
    return [clip(x1), clip(y1), clip(x2), clip(y2)]

"""用像素核对模型说的颜色对不对。零 API 成本。

为什么必须做：颜色是这个数据集里最伤的一类错误。模型说「白色车身的卡车」，
框却落在一辆蓝色卡车上 —— 训练时模型学到的是「白色」对应蓝色卡车。它同时
污染两处：

    ground_attribute  指代靠属性区分同类目标，颜色错了就指向了别的目标（主线 30%）
    attribute_qa      问「这个区域是什么颜色」，直接答错

而颜色恰恰是**能从像素里客观量出来的**，不该靠模型自觉，也不该等质检模型
主观打分 —— 那还是拿模型查模型。

判据刻意宽松，**只在明显冲突时才否定**：
  - 说无彩色（白/黑/灰/银）却量到高饱和度 -> 冲突
  - 说某个色相却量到完全对不上的色相，且饱和度足够高 -> 冲突
  - 其余一律放行

宽松是有意的。航拍小目标、阴影、JPEG 压缩都会让颜色发飘，卡太严会把大量
正确样本误杀。这道闸的定位是「拦住离谱的」，不是「精确判定颜色」。
"""

from __future__ import annotations

import colorsys
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# 无彩色词：靠饱和度判，不看色相
ACHROMATIC = {"白", "黑", "灰", "银"}

# 有彩色词 -> 色相区间（HSV 的 H，0~360）。跨 0 度的红色写成两段。
HUE_BANDS: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "红": ((0, 20), (340, 360)),
    "橙": ((15, 45),),
    "黄": ((40, 70),),
    "绿": ((70, 165),),
    "青": ((165, 200),),
    "蓝": ((195, 260),),
    "紫": ((255, 300),),
    "粉": ((300, 350),),
    "棕": ((10, 45),),          # 棕 = 低明度的橙，色相同橙，靠明度区分
    "褐": ((10, 45),),
    "金": ((35, 60),),
}

# 高于这个饱和度就认为「明显有颜色」，说白/黑/灰/银就站不住
SAT_CHROMATIC = 0.45
# 低于这个饱和度就认为「基本没颜色」，说红/蓝/绿就站不住
SAT_ACHROMATIC = 0.12


def color_word(text: str) -> Optional[str]:
    """从「银灰色」「深蓝色」「白色车身」里抠出颜色字。抠不到返回 None ——
    那说明这不是一个颜色说法（「车头朝左」「停在路边」），不归这道闸管。"""
    if not text:
        return None
    for w in text:
        if w in ACHROMATIC or w in HUE_BANDS:
            return w
    return None


def _median(values):
    v = sorted(values)
    return v[len(v) // 2] if v else 0.0


def sample_hsv(image_path: Path, bbox_px: Sequence[int],
               max_side: int = 24) -> Optional[Tuple[float, float, float]]:
    """取框内中心区域的 HSV 中位数。读不到图返回 None（不判定，放行）。

    只取中心 60% —— 框边缘常常带进背景（路面、天空），
    整框取平均会被背景拉偏。
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            x1, y1, x2, y2 = (int(v) for v in bbox_px)
            w, h = x2 - x1, y2 - y1
            if w < 4 or h < 4:
                return None            # 太小了，量不准，放行
            dx, dy = int(w * 0.2), int(h * 0.2)
            crop = im.crop((x1 + dx, y1 + dy, x2 - dx, y2 - dy))
            if crop.width < 2 or crop.height < 2:
                crop = im.crop((x1, y1, x2, y2))
            crop.thumbnail((max_side, max_side))
            pixels = list(crop.getdata())
    except (OSError, ValueError) as exc:
        logger.debug("读图取色失败 %s：%s", image_path, exc)
        return None
    if not pixels:
        return None
    hsv = [colorsys.rgb_to_hsv(r / 255, g / 255, b / 255) for r, g, b in pixels]
    return (_median([p[0] for p in hsv]) * 360,
            _median([p[1] for p in hsv]),
            _median([p[2] for p in hsv]))


def conflicts(claimed: str, hsv: Optional[Tuple[float, float, float]]) -> bool:
    """模型说的颜色和实测像素明显冲突吗。拿不准一律返回 False（放行）。"""
    word = color_word(claimed)
    if word is None or hsv is None:
        return False
    hue, sat, _val = hsv

    if word in ACHROMATIC:
        # 说白/黑/灰/银，却量到明显的颜色
        return sat >= SAT_CHROMATIC

    bands = HUE_BANDS.get(word)
    if not bands:
        return False
    if sat < SAT_ACHROMATIC:
        # 说红/蓝/绿，却量到基本没有饱和度（是个灰白的东西）
        return True
    return not any(lo <= hue <= hi for lo, hi in bands)


def check(image_path: Path, bbox_px: Sequence[int], claimed: str) -> bool:
    """便捷入口：这个颜色说法站得住吗。"""
    return not conflicts(claimed, sample_hsv(image_path, bbox_px))

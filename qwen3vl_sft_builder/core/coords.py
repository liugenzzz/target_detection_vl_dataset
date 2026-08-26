"""坐标换算。YOLO 归一化 -> 像素 -> Qwen3-VL 的 0~1000 整数坐标。

换算逻辑移植自 qweb3vl_grouding_vqa_lp_gai（本项目不 import 它，保持独立部署）。

Qwen3-VL-8B-Instruct 吃 0~1000 归一化坐标，格式 bbox_2d = [x1, y1, x2, y2]，
左上-右下两点式。若要改成 1~1000 起点，改 config 里的 coords.origin，不用动代码。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def yolo_to_pixel_xyxy(cx: float, cy: float, w: float, h: float,
                       img_w: int, img_h: int) -> List[float]:
    """YOLO 的 [cx, cy, w, h]（归一化、中心点）-> 像素 [x1, y1, x2, y2]。"""
    bw, bh = w * img_w, h * img_h
    x1 = cx * img_w - bw / 2
    y1 = cy * img_h - bh / 2
    x1 = clamp(x1, 0.0, float(img_w))
    y1 = clamp(y1, 0.0, float(img_h))
    x2 = clamp(x1 + bw, 0.0, float(img_w))
    y2 = clamp(y1 + bh, 0.0, float(img_h))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def pixel_to_bbox2d(bbox: Sequence[float], img_w: int, img_h: int,
                    scale: int = 1000, origin: int = 0) -> List[int]:
    """像素 [x1,y1,x2,y2] -> Qwen3-VL 的 0~scale 整数坐标。"""
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"图片尺寸必须为正，得到 {img_w}x{img_h}")
    x1, y1, x2, y2 = bbox
    vals = [
        round(clamp(x1, 0.0, img_w) / img_w * scale),
        round(clamp(y1, 0.0, img_h) / img_h * scale),
        round(clamp(x2, 0.0, img_w) / img_w * scale),
        round(clamp(y2, 0.0, img_h) / img_h * scale),
    ]
    if origin:
        vals = [v + origin for v in vals]
    return [int(clamp(v, origin, scale + origin)) for v in vals]


def yolo_to_bbox2d(cx: float, cy: float, w: float, h: float,
                   img_w: int, img_h: int,
                   scale: int = 1000, origin: int = 0) -> List[int]:
    """一步到位：YOLO 归一化 -> Qwen3-VL 0~scale 坐标。

    注意这里不经过像素中转也能算，但仍走一遍像素，是为了让越界裁剪
    在像素域完成，和标注软件的行为一致。
    """
    return pixel_to_bbox2d(
        yolo_to_pixel_xyxy(cx, cy, w, h, img_w, img_h), img_w, img_h, scale, origin
    )


def zone_of(cx: float, cy: float) -> str:
    """3x3 空间分区，用于生成模板空间指代短语。"""
    horiz = "左" if cx < 1 / 3 else "右" if cx > 2 / 3 else "中"
    vert = "上" if cy < 1 / 3 else "下" if cy > 2 / 3 else "中"
    return vert + horiz

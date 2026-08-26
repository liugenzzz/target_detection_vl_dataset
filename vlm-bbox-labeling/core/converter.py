"""坐标处理：解析模型给的 bbox、修正越界、换算成 YOLO 归一化格式。

这一步全部由代码完成，不交给模型算——模型算归一化中心点很容易出错。
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_bbox(item: dict) -> Optional[List[float]]:
    """从模型返回的一条记录里取出 [x1,y1,x2,y2]。兼容几种常见字段名。"""
    for key in ("bbox_2d", "bbox", "box", "bounding_box", "coordinates", "坐标"):
        if key in item:
            val = item[key]
            if isinstance(val, (list, tuple)) and len(val) == 4:
                try:
                    return [float(v) for v in val]
                except (ValueError, TypeError):
                    return None
            if isinstance(val, dict):
                # {"x1":..,"y1":..,"x2":..,"y2":..} 或 {"x":..,"y":..,"w":..,"h":..}
                if all(k in val for k in ("x1", "y1", "x2", "y2")):
                    try:
                        return [float(val["x1"]), float(val["y1"]), float(val["x2"]), float(val["y2"])]
                    except (ValueError, TypeError):
                        return None
                if all(k in val for k in ("x", "y", "width", "height")):
                    try:
                        x, y = float(val["x"]), float(val["y"])
                        return [x, y, x + float(val["width"]), y + float(val["height"])]
                    except (ValueError, TypeError):
                        return None
    # 平铺写法
    if all(k in item for k in ("x1", "y1", "x2", "y2")):
        try:
            return [float(item["x1"]), float(item["y1"]), float(item["x2"]), float(item["y2"])]
        except (ValueError, TypeError):
            return None
    return None


def normalize_bbox(
    bbox: List[float], img_w: int, img_h: int, mode: str = "auto"
) -> Tuple[Optional[List[float]], List[str]]:
    """把模型给的坐标修正成合法的原图像素坐标 [x1,y1,x2,y2]。

    mode 指定模型返回的坐标格式：
      per_mille = 0~1000 千分比坐标（Qwen3.6 实测就是这种）
      pixel     = 绝对像素坐标
      relative  = 0~1 归一化坐标
      auto      = 自动判断（有明确格式时不要用这个，图片尺寸接近1000时会误判）

    返回 (修正后的坐标, issues)。坐标非法到无法修复时返回 (None, issues)。
    """
    issues = []
    if not bbox or len(bbox) != 4:
        return None, ["bbox 格式非法"]

    x1, y1, x2, y2 = bbox

    if mode == "per_mille":
        x1, y1 = x1 / 1000.0 * img_w, y1 / 1000.0 * img_h
        x2, y2 = x2 / 1000.0 * img_w, y2 / 1000.0 * img_h
    elif mode == "relative":
        x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
    elif mode == "pixel":
        pass
    else:
        x1, y1, x2, y2, auto_issues = _auto_scale(x1, y1, x2, y2, img_w, img_h)
        issues.extend(auto_issues)

    # 保证 x1<x2, y1<y2
    if x1 > x2:
        x1, x2 = x2, x1
        issues.append("x1>x2，已自动交换")
    if y1 > y2:
        y1, y2 = y2, y1
        issues.append("y1>y2，已自动交换")

    # 裁剪到图片范围内
    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(img_w), x2), min(float(img_h), y2)
    if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
        issues.append("坐标超出图片边界，已裁剪")
    x1, y1, x2, y2 = cx1, cy1, cx2, cy2

    if x2 - x1 < 1 or y2 - y1 < 1:
        issues.append("修正后框的宽或高小于1像素，判定为无效框")
        return None, issues

    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)], issues


def _auto_scale(x1, y1, x2, y2, img_w, img_h):
    """自动判断坐标格式（启发式，仅在 COORD_MODE=auto 时使用）。"""
    issues = []
    max_val = max(abs(v) for v in (x1, y1, x2, y2))

    # 分轴判断越界，比只看最大值可靠：
    # 图片本身小于1000像素时，千分比坐标的数值不一定大于图片尺寸。
    x_overflow = max(abs(x1), abs(x2)) > img_w + 1
    y_overflow = max(abs(y1), abs(y2)) > img_h + 1

    if max_val <= 1.001 and img_w > 1 and img_h > 1:
        x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
        issues.append("自动判定为0~1相对坐标，已换算")
    elif (x_overflow or y_overflow) and max_val <= 1000.5:
        x1, y1 = x1 / 1000.0 * img_w, y1 / 1000.0 * img_h
        x2, y2 = x2 / 1000.0 * img_w, y2 / 1000.0 * img_h
        issues.append("自动判定为0~1000千分比坐标，已换算")
        if max(img_w, img_h) > 800:
            issues.append("注意：图片尺寸接近1000，自动判定可能有误，建议显式配置 COORD_MODE")

    return x1, y1, x2, y2, issues


def to_yolo(bbox: List[float], img_w: int, img_h: int) -> List[float]:
    """[x1,y1,x2,y2] 像素坐标 -> YOLO 的 [cx, cy, w, h] 归一化坐标。"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return [round(max(0.0, min(1.0, v)), 6) for v in (cx, cy, w, h)]


def build_yolo_txt(detections: List[dict]) -> str:
    """拼成 YOLO 标注文件内容：每行 `class_id cx cy w h`。"""
    lines = []
    for d in detections:
        if not d.get("valid"):
            continue
        cid = d.get("class_id")
        yolo = d.get("yolo")
        if cid is None or not yolo:
            continue
        lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in yolo))
    return "\n".join(lines)


def iou(box_a: List[float], box_b: List[float]) -> float:
    """两个 [x1,y1,x2,y2] 框的交并比，用于去重。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

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


def dedup(detections: List[dict], iou_thr: float = 0.85) -> List[dict]:
    """同类别、高度重叠的框去重（模型偶尔会把同一个物体标两次）。"""
    kept = []
    for det in detections:
        if not det.get("valid"):
            kept.append(det)
            continue
        dup = False
        for k in kept:
            if not k.get("valid"):
                continue
            if k.get("class_id") == det.get("class_id"):
                if iou(k["bbox_pixel"], det["bbox_pixel"]) >= iou_thr:
                    dup = True
                    break
        if not dup:
            kept.append(det)
    return kept

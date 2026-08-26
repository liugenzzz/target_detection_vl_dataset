"""把模型返回的原始条目，加工成统一的 detection 结构。

三个策略共用这一套后处理，保证返回结构完全一致，方便横向对比。
"""

import logging
from typing import Dict, List, Optional

import config
from core import converter
from core.classes import ClassTable
from core.qwen_client import confidence_for_number

logger = logging.getLogger(__name__)


def pick_class_name(item: dict) -> Optional[str]:
    for key in ("class_name", "label", "name", "类别", "名称", "category"):
        if key in item and item[key] is not None:
            return str(item[key]).strip()
    return None


def pick_class_id(item: dict):
    for key in ("class_id", "id", "编号", "index", "class"):
        if key in item and item[key] is not None:
            return item[key]
    return None


def build_detections(
    raw_items: List[dict],
    table: ClassTable,
    img_w: int,
    img_h: int,
    token_probs=None,
) -> List[Dict]:
    """raw_items -> 标准 detection 列表。

    每条 detection 结构：
    {
      "class_id": 214,
      "class_name": "xxx",
      "bbox_pixel": [x1,y1,x2,y2],
      "yolo": [cx,cy,w,h],
      "confidence": 0.93 或 null,
      "valid": true/false,
      "issues": ["..."],
      "raw": {原始条目}
    }

    valid=False 的条目不会写进 YOLO txt，但仍会保留在返回结果里，
    实验阶段这些"失败样本"本身就是重要数据，不能悄悄丢掉。
    """
    detections = []
    # logprobs 搜索游标：保证同一 class_id 的多个框取到各自位置的 token
    lp_cursor = 0

    for item in raw_items:
        issues = []

        raw_id = pick_class_id(item)
        raw_name = pick_class_name(item)
        class_id, class_name, cls_issues = table.validate(raw_id, raw_name)
        issues.extend(cls_issues)

        bbox_raw = converter.parse_bbox(item)
        if bbox_raw is None:
            issues.append("未能从该条目解析出 bbox 坐标")
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox_pixel": None,
                    "yolo": None,
                    "confidence": None,
                    "valid": False,
                    "issues": issues,
                    "raw": item,
                }
            )
            continue

        bbox, box_issues = converter.normalize_bbox(bbox_raw, img_w, img_h, config.COORD_MODE)
        issues.extend(box_issues)

        # 面积异常检测：覆盖大半张图的框大概率是误检
        if bbox is not None and img_w > 0 and img_h > 0 and config.BIG_BOX_RATIO < 1.0:
            area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / float(img_w * img_h)
            if area_ratio >= config.BIG_BOX_RATIO:
                issues.append(f"该框占整图面积的 {area_ratio:.0%}，可能是误检，请重点核对")

        valid = bbox is not None and class_id is not None
        yolo = converter.to_yolo(bbox, img_w, img_h) if bbox else None

        # 置信度：优先用模型自己给的（不太可靠），有 logprobs 时用 logprobs 覆盖（更可靠）
        conf = item.get("confidence") or item.get("score") or item.get("置信度")
        try:
            conf = round(float(conf), 4) if conf is not None else None
        except (ValueError, TypeError):
            conf = None

        if token_probs and class_id is not None:
            lp_conf, lp_cursor = confidence_for_number(token_probs, str(class_id), lp_cursor)
            if lp_conf is not None:
                conf = lp_conf

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "bbox_pixel": bbox,
                "yolo": yolo,
                "confidence": conf,
                "valid": valid,
                "issues": issues,
                "raw": item,
            }
        )

    detections = converter.dedup(detections)
    detections = detect_repetition(detections, img_w, img_h)
    detections = cap_per_class(detections, config.MAX_BOXES_PER_CLASS)
    return detections


def detect_repetition(detections: List[Dict], img_w: int, img_h: int) -> List[Dict]:
    """检测"重复生成死循环"产生的假框。

    模型卡在循环里时，会吐出一串尺寸几乎相同、位置等距递推的框
    （实测表现为斜向阶梯状或竖直堆叠的一摞）。这类框数值上完全合法，
    坐标校验和置信度都抓不到 —— 循环中的 token 概率反而接近 1.0。

    判据：同一类别中，连续若干个框的宽高高度一致，且中心点间距近似恒定。
    """
    by_class = {}
    for idx, d in enumerate(detections):
        if not d.get("valid") or not d.get("bbox_pixel"):
            continue
        by_class.setdefault(d.get("class_id"), []).append((idx, d))

    for cid, items in by_class.items():
        if len(items) < 5:
            continue

        def size_of(d):
            b = d["bbox_pixel"]
            return (b[2] - b[0], b[3] - b[1])

        def center_of(d):
            b = d["bbox_pixel"]
            return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

        # 在类别内部找"连续的等距同尺寸串"，而不是要求整个类别都一致 ——
        # 一张图里既可能有正常框，也可能有循环产生的假框，两者会混在一起。
        n = len(items)
        i = 0
        while i < n:
            run = [i]
            for j in range(i + 1, n):
                w0, h0 = size_of(items[run[0]][1])
                wj, hj = size_of(items[j][1])
                if w0 <= 0 or h0 <= 0:
                    break
                # 尺寸必须接近
                if abs(wj - w0) / w0 > 0.15 or abs(hj - h0) / h0 > 0.15:
                    break
                run.append(j)

            if len(run) >= 5:
                cs = [center_of(items[k][1]) for k in run]
                steps = []
                for a in range(len(cs) - 1):
                    dx = cs[a + 1][0] - cs[a][0]
                    dy = cs[a + 1][1] - cs[a][1]
                    steps.append((dx ** 2 + dy ** 2) ** 0.5)
                steps = [s for s in steps if s > 0.5]

                if len(steps) >= 4:
                    m = sum(steps) / len(steps)
                    # 间距近似恒定 = 等距递推，典型的循环生成特征
                    if m > 0 and (max(steps) - min(steps)) / m <= 0.3:
                        for k in run:
                            d = items[k][1]
                            d["valid"] = False
                            d.setdefault("issues", []).append(
                                f"疑似模型重复生成：检测到 {len(run)} 个等距同尺寸框连续排列，已判为无效"
                            )
                i = run[-1] + 1
            else:
                i += 1

    return detections


def cap_per_class(detections: List[Dict], limit: int) -> List[Dict]:
    """单个类别的框数量上限，超出的判为无效（防重复生成刷屏）。"""
    if limit <= 0:
        return detections
    counter = {}
    for d in detections:
        if not d.get("valid"):
            continue
        cid = d.get("class_id")
        counter[cid] = counter.get(cid, 0) + 1
        if counter[cid] > limit:
            d["valid"] = False
            d.setdefault("issues", []).append(
                f"该类别框数超过上限 {limit}，已判为无效（可调 MAX_BOXES_PER_CLASS）"
            )
    return detections


def build_stats(detections: List[Dict]) -> Dict:
    valid = [d for d in detections if d.get("valid")]
    flagged = [d for d in detections if d.get("issues")]
    class_ids = sorted({d["class_id"] for d in valid if d.get("class_id") is not None})
    return {
        "total": len(detections),
        "valid": len(valid),
        "invalid": len(detections) - len(valid),
        "flagged_for_review": len(flagged),
        "distinct_classes": len(class_ids),
        "class_ids": class_ids,
    }

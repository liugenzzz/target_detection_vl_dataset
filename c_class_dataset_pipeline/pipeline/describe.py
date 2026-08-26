"""生成"描述语句"——招标指标里 C 类数据集要求每条样本包含
"图像、文本指标、标注区域/答案、描述语句"四部分。

前两步（vlm-bbox-labeling 打框 + qweb3vl_grouding_vqa_lp_gai 转换）已经产出了
"文本指标"（指令）和"标注区域/答案"（bbox_2d JSON）。这个模块补上第四部分：
针对每条样本实际引用到的目标框，生成一句自然语言描述。

当前是规则模板生成（确定性、可复现、不依赖任何外部模型调用）。
如果之后要接真实的 VLM/LLM 生成更自然的描述，只需要替换 describe_boxes()
的实现，上层 schema.py 不用改。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .deps import GroundingBox, GroundingRecord, reference_phrase, spatial_phrase

SMALL_BOX_AREA_RATIO = 0.01  # 框面积占全图比例低于此值，提示"尺寸较小，注意复核"


def _size_hint(box: GroundingBox, record: GroundingRecord) -> str:
    x1, y1, x2, y2 = box.bbox_pixel
    area_ratio = (x2 - x1) * (y2 - y1) / max(record.width * record.height, 1)
    if area_ratio < SMALL_BOX_AREA_RATIO:
        return "，目标尺寸较小，建议重点复核"
    return ""


def resolve_target_boxes(record: GroundingRecord, metadata: Dict[str, Any]) -> List[GroundingBox]:
    """从样本 metadata 反推这条样本实际指代的目标框列表。

    对应 qweb3vl_grouding_vqa_lp_gai/yolo_visual_grounding_sft.py 里
    build_sft_samples() 四种任务类型写入 metadata 的方式：
      - vlm_grounding_qa 系列：metadata["target_indices"]
      - ground_single：metadata["box_index"]
      - detect_label：metadata["label"]
      - detect_all：不带额外筛选字段，代表全部框
    """
    by_index = {box.index: box for box in record.boxes}

    if "target_indices" in metadata:
        return [by_index[i] for i in metadata["target_indices"] if i in by_index]
    if "box_index" in metadata:
        box = by_index.get(metadata["box_index"])
        return [box] if box is not None else []
    if "label" in metadata:
        return [box for box in record.boxes if box.label == metadata["label"]]
    return list(record.boxes)


def describe_boxes(boxes: Sequence[GroundingBox], record: GroundingRecord) -> str:
    """给定一组目标框，生成一句中文描述语句。"""
    if not boxes:
        return "图中未定位到符合条件的目标。"

    if len(boxes) == 1:
        box = boxes[0]
        return f"{spatial_phrase(box, record)}，类别为“{box.label}”{_size_hint(box, record)}。"

    labels = [box.label for box in boxes]
    distinct_labels = sorted(set(labels), key=labels.index)

    if len(distinct_labels) == 1:
        label = distinct_labels[0]
        boxes_by_label = {label: list(boxes)}
        phrases = [reference_phrase(box, boxes_by_label[label], record) for box in boxes]
        hint = "，其中包含尺寸较小的目标，建议重点复核" if any(
            (b.bbox_pixel[2] - b.bbox_pixel[0]) * (b.bbox_pixel[3] - b.bbox_pixel[1])
            / max(record.width * record.height, 1) < SMALL_BOX_AREA_RATIO
            for b in boxes
        ) else ""
        return f"图中共有 {len(boxes)} 个“{label}”，分别为{'、'.join(phrases)}{hint}。"

    counts = {label: labels.count(label) for label in distinct_labels}
    parts = "、".join(f"{label}（{count}个）" for label, count in counts.items())
    return f"图中共标注了 {len(boxes)} 个目标，涉及 {len(distinct_labels)} 个类别：{parts}。"


def attach_description(sample: Dict[str, Any], record: GroundingRecord) -> Dict[str, Any]:
    """给一条 qweb3vl 生成的 SFT 样本补上 `description` 字段（原地返回新 dict，不修改入参）。"""
    metadata = sample.get("metadata", {})
    boxes = resolve_target_boxes(record, metadata)
    sample = dict(sample)
    sample["description"] = describe_boxes(boxes, record)
    return sample

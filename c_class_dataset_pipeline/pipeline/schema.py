"""C 类数据集的最终样本结构，对应招标指标原文：

    C 类：图中目标标注数据集
    用于训练模型依据指标描述自动标注图中目标、回答图像相关问题；
    每条样本包含图像、文本指标、标注区域/答案、描述语句；
    数量 ≥10 万份

四个必须字段 <-> 本项目字段的对应关系：

    图像         -> image
    文本指标      -> instruction   （对应"依据指标描述自动标注/回答问题"里的"指标描述"）
    标注区域/答案  -> answer        （bbox_2d + label，来自 qweb3vl_grouding_vqa_lp_gai）
    描述语句      -> description   （本项目 describe.py 新增）

同时保留 `conversations`（ShareGPT 格式）和 `metadata`，
这样样本既满足指标要求，也能不做二次转换直接喂给 Qwen3-VL 做 SFT 训练。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

REQUIRED_FIELDS = ("image", "instruction", "answer", "description")


def _instruction_from_conversations(conversations: List[Dict[str, str]]) -> str:
    for turn in conversations:
        if turn.get("from") == "human":
            value = turn.get("value", "")
            return value[len("<image>\n"):] if value.startswith("<image>\n") else value
    return ""


def _answer_from_conversations(conversations: List[Dict[str, str]]) -> Any:
    for turn in conversations:
        if turn.get("from") == "gpt":
            try:
                return json.loads(turn.get("value", "null"))
            except json.JSONDecodeError:
                return turn.get("value")
    return None


def to_c_class_record(sample: Dict[str, Any]) -> Dict[str, Any]:
    """把 qweb3vl 样本（已由 describe.attach_description 补上 description）
    整理成指标要求的最终结构。"""
    images = sample.get("images") or ([sample["image"]] if "image" in sample else [])
    conversations = sample.get("conversations", [])
    metadata = sample.get("metadata", {})

    return {
        "id": sample.get("id"),
        "image": images[0] if images else None,
        "instruction": _instruction_from_conversations(conversations),
        "answer": _answer_from_conversations(conversations),
        "description": sample.get("description", ""),
        "task_type": metadata.get("task_type"),
        "conversations": conversations,
        "metadata": metadata,
    }


def validate_c_class_record(record: Dict[str, Any]) -> List[str]:
    """校验一条样本是否满足指标要求的四要素，返回问题列表（空列表=合格）。"""
    issues = []
    if not record.get("image"):
        issues.append("缺少 image 字段")
    if not record.get("instruction"):
        issues.append("缺少 instruction（文本指标）字段")
    if record.get("answer") in (None, [], {}):
        issues.append("缺少 answer（标注区域/答案）字段")
    if not record.get("description"):
        issues.append("缺少 description（描述语句）字段")
    return issues

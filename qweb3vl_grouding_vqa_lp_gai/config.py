"""Configuration and prompts for YOLO-like visual grounding SFT generation."""

from __future__ import annotations

from typing import Any, Dict


DEFAULT_BBOX_SCALE = 1000
DEFAULT_COORDINATE_MODE = "qwen_relative_1000"
DEFAULT_DATASET_FORMAT = "sharegpt"
DEFAULT_TASK_TYPES = ["ground_single"]

SYSTEM_PROMPT = (
    "你是一个视觉定位SFT数据标注助手。你只能基于图片和给定的真实标注框，"
    "生成自然、简洁的指代表达和定位问题。不要修改、推断或新增bbox坐标。"
)

DETECT_ALL_PROMPT = "请定位图中所有已标注目标，输出JSON数组，每个元素包含bbox_2d和label。"
DETECT_LABEL_PROMPT_TEMPLATE = "请定位图中所有“{label}”，输出JSON数组，每个元素包含bbox_2d和label。"
GROUND_SINGLE_PROMPT_TEMPLATE = "请定位{phrase}，输出一个包含bbox_2d和label的JSON对象。"

DEFAULT_CONFIG: Dict[str, Any] = {
    "annotation_dir": "yolo_samples",
    "output_jsonl": "output/qwen3vl_yolo_grounding_sft.jsonl",
    "image_dir": "",
    "annotation_glob": "*.json",
    "recursive": False,
    "generation": {
        "coordinate_mode": DEFAULT_COORDINATE_MODE,
        "bbox_scale": DEFAULT_BBOX_SCALE,
        "dataset_format": DEFAULT_DATASET_FORMAT,
        "output_image_path": "filename",
        "task_types": DEFAULT_TASK_TYPES,
        "qa_per_image_min": 3,
        "qa_per_image_max": 5,
        "include_metadata": True,
        "ignore_labels": ["__ignore__"],
        "min_box_size": 1,
    },
    "prompt": {
        "system": SYSTEM_PROMPT,
        "detect_all": DETECT_ALL_PROMPT,
        "detect_label_template": DETECT_LABEL_PROMPT_TEMPLATE,
        "ground_single_template": GROUND_SINGLE_PROMPT_TEMPLATE,
    },
}


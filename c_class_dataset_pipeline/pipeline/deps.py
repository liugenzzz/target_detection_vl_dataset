"""把仓库里另外两个项目挂到 sys.path 上，做到"直接复用代码、不复制粘贴"。

两个兄弟项目：

    vlm-bbox-labeling/              -- 第一步：调用 VLM 给图片打预标注框（YOLO 格式）
    qweb3vl_grouding_vqa_lp_gai/    -- 第二步：把框转换成 Qwen3-VL 视觉定位 SFT 语料

本项目（c_class_dataset_pipeline）是第三步，负责把前两步串起来，
再补一道"描述语句"生成，最后拼成招标指标里 C 类数据集要求的样本结构。

只在这里做一次路径拼接，其余模块统一从这里导入，避免到处 sys.path.insert。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BBOX_LABELING_DIR = REPO_ROOT / "vlm-bbox-labeling"
GROUNDING_VQA_DIR = REPO_ROOT

for p in (BBOX_LABELING_DIR, GROUNDING_VQA_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# --- 来自 vlm-bbox-labeling：类别表加载（编号<->名称双向校验） ---
from core.classes import ClassTable, load_class_table  # noqa: E402

# --- 来自 qweb3vl_grouding_vqa_lp_gai：坐标换算、样本生成的核心逻辑 ---
from qweb3vl_grouding_vqa_lp_gai.yolo_visual_grounding_sft import (  # noqa: E402
    DEFAULT_CONFIG,
    GroundingBox,
    GroundingRecord,
    build_record_from_payload,
    build_sft_samples,
    deep_merge,
    read_image_size,
    reference_phrase,
    safe_id_component,
    spatial_phrase,
    unique_labels,
)

__all__ = [
    "REPO_ROOT",
    "BBOX_LABELING_DIR",
    "GROUNDING_VQA_DIR",
    "ClassTable",
    "load_class_table",
    "DEFAULT_CONFIG",
    "GroundingBox",
    "GroundingRecord",
    "build_record_from_payload",
    "build_sft_samples",
    "deep_merge",
    "read_image_size",
    "reference_phrase",
    "safe_id_component",
    "spatial_phrase",
    "unique_labels",
]

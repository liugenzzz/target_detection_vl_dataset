"""管道编排：标注 payload -> (qweb3vl 转换) -> (补描述语句) -> C 类样本 -> 写 JSONL。

典型用法见 scripts/run_on_coco128.py（联调用，走 yolo_gt_adapter）
和 scripts/run_production.py（生产用，走 bbox_service_adapter）。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .deps import DEFAULT_CONFIG, build_record_from_payload, build_sft_samples, deep_merge
from .describe import attach_description
from .schema import to_c_class_record, validate_c_class_record

# 生产目标：招标指标 C 类要求 ≥10 万条样本。
TARGET_SAMPLE_COUNT = 100_000

# 尽量把 4 种任务类型都打开，最大化单张图能摊出的样本数：
#   detect_all       每图 1 条（全部目标）
#   detect_label      每类别 1 条
#   ground_single     每个框 1 条（单目标指代定位）
#   vlm_grounding_qa  每图 qa_per_image_min~max 条（综合问答，覆盖"回答图像相关问题"）
DEFAULT_GENERATION_OVERRIDE: Dict[str, Any] = {
    "generation": {
        "task_types": ["detect_all", "detect_label", "ground_single", "vlm_grounding_qa"],
        "include_metadata": True,  # describe.py 依赖 metadata 反推目标框，不能关
        "qa_per_image_min": 3,
        "qa_per_image_max": 5,
    }
}


def build_records_for_payload(
    payload: Dict[str, Any],
    generation_override: Dict[str, Any] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """单条标注 payload -> 该图片对应的全部 C 类样本列表，附带该图有效框数。"""
    config = deep_merge(DEFAULT_CONFIG, generation_override or DEFAULT_GENERATION_OVERRIDE)
    record = build_record_from_payload(payload, annotation_path=str(payload.get("id") or "sample"), config=config)
    if not record.boxes:
        return [], 0

    samples = build_sft_samples(record, config)
    c_class_records = [to_c_class_record(attach_description(s, record)) for s in samples]
    return c_class_records, len(record.boxes)


def build_dataset(
    payloads: Iterable[Dict[str, Any]],
    output_jsonl: Path,
    generation_override: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """跑完整批 payload，写 JSONL，返回统计信息（含扩量到 10 万条的估算）。"""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_boxes = 0
    task_type_counter: Counter = Counter()
    invalid_records = 0
    total_samples = 0

    with output_jsonl.open("w", encoding="utf-8") as fh:
        for payload in payloads:
            total_images += 1
            records, box_count = build_records_for_payload(payload, generation_override)
            total_boxes += box_count
            for rec in records:
                issues = validate_c_class_record(rec)
                if issues:
                    invalid_records += 1
                    continue
                task_type_counter[rec.get("task_type")] += 1
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_samples += 1

    avg_samples_per_image = round(total_samples / total_images, 2) if total_images else 0.0
    images_needed_for_target = (
        int(-(-TARGET_SAMPLE_COUNT // avg_samples_per_image)) if avg_samples_per_image > 0 else None
    )

    return {
        "output_jsonl": str(output_jsonl),
        "images_processed": total_images,
        "boxes_total": total_boxes,
        "samples_total": total_samples,
        "samples_by_task_type": dict(task_type_counter),
        "invalid_records_dropped": invalid_records,
        "avg_samples_per_image": avg_samples_per_image,
        "target_sample_count": TARGET_SAMPLE_COUNT,
        "meets_target": total_samples >= TARGET_SAMPLE_COUNT,
        "images_needed_to_reach_target_at_this_ratio": images_needed_for_target,
    }


def print_report(stats: Dict[str, Any]) -> None:
    print("=" * 60)
    print(f"输出文件         : {stats['output_jsonl']}")
    print(f"处理图片数        : {stats['images_processed']}")
    print(f"有效标注框数      : {stats['boxes_total']}")
    print(f"生成样本数        : {stats['samples_total']}")
    print(f"按任务类型分布    : {stats['samples_by_task_type']}")
    if stats["invalid_records_dropped"]:
        print(f"因缺字段被丢弃    : {stats['invalid_records_dropped']}")
    print(f"平均每图样本数     : {stats['avg_samples_per_image']}")
    print(f"指标要求 (C 类)   : ≥ {stats['target_sample_count']}")
    if stats["meets_target"]:
        print("结论             : 已达标 ✅")
    else:
        print(
            "结论             : 未达标 ❌ —— 按当前平均每图样本数估算，"
            f"需要约 {stats['images_needed_to_reach_target_at_this_ratio']} 张源图片才能凑够 "
            f"{stats['target_sample_count']} 条样本（见 scripts/estimate_scale.py）。"
        )
    print("=" * 60)

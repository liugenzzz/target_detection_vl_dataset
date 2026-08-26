#!/usr/bin/env python
"""Build Qwen3-VL visual grounding SFT data from x/y/width/height boxes."""

from __future__ import annotations

import argparse
import copy
import json
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .config import DEFAULT_CONFIG
except ImportError:  # pragma: no cover - supports direct script execution.
    from config import DEFAULT_CONFIG


@dataclass
class GroundingBox:
    index: int
    label: str
    bbox_pixel: List[float]
    bbox_2d: List[int]


@dataclass
class GroundingQAPair:
    question: str
    target_indices: List[int]
    qa_type: str = "referring"


@dataclass
class GroundingRecord:
    annotation_path: str
    image_path: str
    width: int
    height: int
    boxes: List[GroundingBox]
    qa_pairs: List[GroundingQAPair] = field(default_factory=list)


def deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def clean_label(label: Any) -> str:
    return str(label or "").strip()


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_xywh_points(points: Dict[str, Any], width: int, height: int) -> List[float]:
    """Convert top-left xywh pixel box to clipped [x1, y1, x2, y2]."""
    try:
        x = float(points["x"])
        y = float(points["y"])
        box_width = float(points["width"])
        box_height = float(points["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("points must contain numeric x, y, width and height") from exc

    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Box width and height must be positive, got {box_width}x{box_height}")

    x1 = clamp(x, 0.0, float(width))
    y1 = clamp(y, 0.0, float(height))
    x2 = clamp(x + box_width, 0.0, float(width))
    y2 = clamp(y + box_height, 0.0, float(height))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def pixel_box_to_qwen_bbox(
    bbox: Sequence[float],
    width: int,
    height: int,
    scale: int = 1000,
) -> List[int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {width}x{height}")
    x1, y1, x2, y2 = bbox
    values = [
        round(clamp(x1, 0.0, width) / width * scale),
        round(clamp(y1, 0.0, height) / height * scale),
        round(clamp(x2, 0.0, width) / width * scale),
        round(clamp(y2, 0.0, height) / height * scale),
    ]
    return [int(clamp(value, 0, scale)) for value in values]


def pixel_box_to_integer_bbox(bbox: Sequence[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    return [
        int(round(clamp(x1, 0.0, width))),
        int(round(clamp(y1, 0.0, height))),
        int(round(clamp(x2, 0.0, width))),
        int(round(clamp(y2, 0.0, height))),
    ]


def convert_bbox(
    bbox: Sequence[float],
    width: int,
    height: int,
    coordinate_mode: str,
    scale: int,
) -> List[int]:
    if coordinate_mode == "pixel":
        return pixel_box_to_integer_bbox(bbox, width, height)
    if coordinate_mode == "qwen_relative_1000":
        return pixel_box_to_qwen_bbox(bbox, width, height, scale)
    raise ValueError(f"Unsupported coordinate_mode: {coordinate_mode}")


def read_png_size(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    return None


def read_jpeg_size(data: bytes) -> Optional[Tuple[int, int]]:
    if not data.startswith(b"\xff\xd8"):
        return None
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        while marker == 0xFF and index < len(data):
            marker = data[index]
            index += 1
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            return None
        segment_length = struct.unpack(">H", data[index : index + 2])[0]
        if segment_length < 2 or index + segment_length > len(data):
            return None
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            height, width = struct.unpack(">HH", data[index + 3 : index + 7])
            return int(width), int(height)
        index += segment_length
    return None


def read_image_size(image_path: Path) -> Tuple[int, int]:
    data = image_path.read_bytes()
    size = read_png_size(data) or read_jpeg_size(data)
    if size is None:
        raise ValueError(f"Unable to read image size from {image_path}")
    return size


def resolve_image_path(annotation_path: Path, image_path: str, image_root: Optional[Path] = None) -> Path:
    raw = Path(image_path)
    if raw.is_absolute():
        return raw
    if image_root is not None:
        return (image_root / raw).resolve()
    return (annotation_path.parent / raw).resolve()


def parse_image_size(payload: Dict[str, Any]) -> Tuple[int, int]:
    width = payload.get("image_width", payload.get("imageWidth", 0))
    height = payload.get("image_height", payload.get("imageHeight", 0))
    try:
        return int(width or 0), int(height or 0)
    except (TypeError, ValueError):
        return 0, 0


def is_ignored_shape(shape: Dict[str, Any], ignore_labels: Sequence[str]) -> bool:
    label = clean_label(shape.get("label"))
    flags = shape.get("flags") or {}
    return label in set(ignore_labels) or bool(flags.get("__ignore__") or flags.get("ignore"))


def build_record_from_payload(
    payload: Dict[str, Any],
    annotation_path: Path | str = "",
    image_root: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> GroundingRecord:
    config_data = deep_merge(DEFAULT_CONFIG, config)
    generation = config_data["generation"]
    annotation = Path(annotation_path) if annotation_path else Path(str(payload.get("id") or "inline_annotation"))
    image_path = str(payload.get("image") or payload.get("imagePath") or annotation.with_suffix(".jpg").name)
    width, height = parse_image_size(payload)
    if width <= 0 or height <= 0:
        width, height = read_image_size(resolve_image_path(annotation, image_path, image_root))

    boxes: List[GroundingBox] = []
    ignore_labels = generation.get("ignore_labels") or []
    min_box_size = float(generation.get("min_box_size", 1))
    for shape_index, shape in enumerate(payload.get("shapes") or []):
        if not isinstance(shape, dict) or is_ignored_shape(shape, ignore_labels):
            continue
        shape_type = shape.get("shape_type")
        if shape_type and shape_type not in {"rectangle", "bbox", "xywh"}:
            continue
        label = clean_label(shape.get("label"))
        if not label:
            continue
        points = shape.get("points")
        if not isinstance(points, dict):
            raise ValueError("YOLO-like shape.points must be {'x', 'y', 'width', 'height'}")
        bbox_pixel = normalize_xywh_points(points, width, height)
        if bbox_pixel[2] - bbox_pixel[0] < min_box_size or bbox_pixel[3] - bbox_pixel[1] < min_box_size:
            continue
        boxes.append(
            GroundingBox(
                index=shape_index,
                label=label,
                bbox_pixel=bbox_pixel,
                bbox_2d=convert_bbox(
                    bbox_pixel,
                    width,
                    height,
                    str(generation.get("coordinate_mode", "qwen_relative_1000")),
                    int(generation.get("bbox_scale", 1000)),
                ),
            )
        )

    return GroundingRecord(
        annotation_path=str(annotation),
        image_path=image_path.replace("\\", "/"),
        width=width,
        height=height,
        boxes=boxes,
    )


def load_record(annotation_path: Path | str, image_root: Optional[Path] = None, config: Optional[Dict[str, Any]] = None) -> GroundingRecord:
    annotation = Path(annotation_path)
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    return build_record_from_payload(payload, annotation_path=annotation, image_root=image_root, config=config)


def safe_id_component(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", Path(str(value)).stem)
    return cleaned.strip("_") or "sample"


def unique_labels(boxes: Sequence[GroundingBox]) -> List[str]:
    labels: List[str] = []
    for box in boxes:
        if box.label not in labels:
            labels.append(box.label)
    return labels


def box_to_answer(box: GroundingBox) -> Dict[str, Any]:
    return {"bbox_2d": box.bbox_2d, "label": box.label}


def answer_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def spatial_phrase(box: GroundingBox, record: GroundingRecord) -> str:
    x1, y1, x2, y2 = box.bbox_pixel
    center_x = (x1 + x2) / 2 / record.width
    center_y = (y1 + y2) / 2 / record.height
    horizontal = "左侧" if center_x < 0.33 else "右侧" if center_x > 0.67 else "中间"
    vertical = "上方" if center_y < 0.33 else "下方" if center_y > 0.67 else "中部"
    return f"图中{vertical}{horizontal}的{box.label}"


def reference_phrase(box: GroundingBox, same_label_boxes: Sequence[GroundingBox], record: GroundingRecord) -> str:
    base = spatial_phrase(box, record)
    if len(same_label_boxes) <= 1:
        return base
    ordered = sorted(same_label_boxes, key=lambda item: (item.bbox_2d[1], item.bbox_2d[0], item.index))
    return f"{base}（第{ordered.index(box) + 1}个{box.label}）"


def clamp_pair_count(value: Any, default: int, lower: int = 1, upper: int = 20) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(upper, parsed))


def qa_count_bounds(config: Dict[str, Any]) -> Tuple[int, int]:
    generation = config.get("generation", {})
    min_pairs = clamp_pair_count(generation.get("qa_per_image_min"), 3)
    max_pairs = clamp_pair_count(generation.get("qa_per_image_max"), 5)
    return min_pairs, max(min_pairs, max_pairs)


def build_fallback_qa_pairs(record: GroundingRecord, config: Dict[str, Any]) -> List[GroundingQAPair]:
    if not record.boxes:
        return []
    _, max_pairs = qa_count_bounds(config)
    pairs = [
        GroundingQAPair(
            question=str(config["prompt"]["detect_all"]),
            target_indices=[box.index for box in record.boxes],
            qa_type="all_objects",
        )
    ]
    for label in unique_labels(record.boxes):
        pairs.append(
            GroundingQAPair(
                question=str(config["prompt"]["detect_label_template"]).format(label=label),
                target_indices=[box.index for box in record.boxes if box.label == label],
                qa_type="category",
            )
        )
        if len(pairs) >= max_pairs:
            return pairs[:max_pairs]

    boxes_by_label = {label: [box for box in record.boxes if box.label == label] for label in unique_labels(record.boxes)}
    for box in record.boxes:
        pairs.append(
            GroundingQAPair(
                question=str(config["prompt"]["ground_single_template"]).format(
                    phrase=reference_phrase(box, boxes_by_label[box.label], record)
                ),
                target_indices=[box.index],
                qa_type="ground_single",
            )
        )
        if len(pairs) >= max_pairs:
            return pairs[:max_pairs]
    return pairs[:max_pairs]


def answers_for_indices(record: GroundingRecord, target_indices: Sequence[int]) -> Any:
    by_index = {box.index: box for box in record.boxes}
    answers = [box_to_answer(by_index[index]) for index in target_indices if index in by_index]
    return answers[0] if len(answers) == 1 else answers


def image_value_for_output(record: GroundingRecord, config: Dict[str, Any], image_path: Optional[Path] = None) -> str:
    style = config.get("generation", {}).get("output_image_path", "filename")
    if style == "absolute":
        return str(image_path.resolve()) if image_path else str(Path(record.image_path).resolve())
    if style == "relative" and image_path:
        base_dir = Path(str(config.get("_config_dir") or ".")).resolve()
        try:
            return str(image_path.resolve().relative_to(base_dir)).replace("\\", "/")
        except ValueError:
            return str(image_path.resolve()).replace("\\", "/")
    return Path(record.image_path).name


def make_sample(
    sample_id: str,
    image_value: str,
    question: str,
    answer: Any,
    metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    sample: Dict[str, Any] = {
        "id": sample_id,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer_json(answer)},
        ],
    }
    if config.get("generation", {}).get("dataset_format", "sharegpt") == "sharegpt":
        sample["images"] = [image_value]
    else:
        sample["image"] = image_value
    if config.get("generation", {}).get("include_metadata", True):
        sample["metadata"] = metadata
    return sample


def build_sft_samples(record: GroundingRecord, config: Dict[str, Any], image_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    task_types = config.get("generation", {}).get("task_types") or ["vlm_grounding_qa"]
    image_value = image_value_for_output(record, config, image_path)
    base_id = safe_id_component(str(record.annotation_path or record.image_path))
    metadata = {
        "source_annotation": record.annotation_path,
        "source_image": record.image_path,
        "image_width": record.width,
        "image_height": record.height,
        "coordinate_mode": config["generation"].get("coordinate_mode", "qwen_relative_1000"),
        "bbox_scale": int(config["generation"].get("bbox_scale", 1000)),
    }
    samples: List[Dict[str, Any]] = []

    if "vlm_grounding_qa" in task_types and record.boxes:
        for index, pair in enumerate(build_fallback_qa_pairs(record, config)):
            answer = answers_for_indices(record, pair.target_indices)
            if answer:
                samples.append(
                    make_sample(
                        f"{base_id}_vlm_grounding_qa_{index}",
                        image_value,
                        pair.question,
                        answer,
                        dict(metadata, task_type=pair.qa_type, target_indices=pair.target_indices),
                        config,
                    )
                )

    if "detect_all" in task_types and record.boxes:
        samples.append(
            make_sample(
                f"{base_id}_detect_all",
                image_value,
                str(config["prompt"]["detect_all"]),
                [box_to_answer(box) for box in record.boxes],
                dict(metadata, task_type="detect_all"),
                config,
            )
        )

    if "detect_label" in task_types:
        for label in unique_labels(record.boxes):
            boxes = [box for box in record.boxes if box.label == label]
            samples.append(
                make_sample(
                    f"{base_id}_detect_label_{safe_id_component(label)}",
                    image_value,
                    str(config["prompt"]["detect_label_template"]).format(label=label),
                    [box_to_answer(box) for box in boxes],
                    dict(metadata, task_type="detect_label", label=label),
                    config,
                )
            )

    if "ground_single" in task_types:
        boxes_by_label = {label: [box for box in record.boxes if box.label == label] for label in unique_labels(record.boxes)}
        for box in record.boxes:
            question = str(config["prompt"]["ground_single_template"]).format(
                phrase=reference_phrase(box, boxes_by_label[box.label], record)
            )
            samples.append(
                make_sample(
                    f"{base_id}_ground_single_{box.index}",
                    image_value,
                    question,
                    box_to_answer(box),
                    dict(metadata, task_type="ground_single", label=box.label, box_index=box.index),
                    config,
                )
            )
    return samples


def convert_annotation_payload(
    payload: Dict[str, Any],
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    config = deep_merge(DEFAULT_CONFIG, None)
    config["_config_dir"] = str(Path.cwd())
    provided_image_path = Path(image_path).resolve() if image_path else None
    record = build_record_from_payload(
        payload,
        annotation_path=str(payload.get("id") or "inline_annotation"),
        image_root=provided_image_path.parent if provided_image_path else None,
        config=config,
    )
    samples = build_sft_samples(record, config, image_path=provided_image_path)
    return {
        "stats": {"annotations": 1, "boxes": len(record.boxes), "samples": len(samples)},
        "samples": samples,
    }


def discover_annotation_files(annotation_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    iterator = annotation_dir.rglob(pattern) if recursive else annotation_dir.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_json_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run(config_path: Optional[Path | str] = None, dry_run: bool = False) -> Dict[str, int]:
    config_file = Path(config_path).resolve() if config_path else None
    config = deep_merge(DEFAULT_CONFIG, load_json_config(config_file) if config_file else None)
    config["_config_dir"] = str(config_file.parent if config_file else Path.cwd())

    base_dir = Path(config["_config_dir"])
    annotation_dir = Path(str(config.get("annotation_dir", "yolo_samples")))
    if not annotation_dir.is_absolute():
        annotation_dir = base_dir / annotation_dir
    output_path = Path(str(config.get("output_jsonl", "grounding_sft.jsonl")))
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    image_root_value = str(config.get("image_dir") or "")
    image_root = base_dir / image_root_value if image_root_value else None

    annotation_files = discover_annotation_files(
        annotation_dir,
        str(config.get("annotation_glob") or "*.json"),
        bool(config.get("recursive", False)),
    )
    samples: List[Dict[str, Any]] = []
    box_count = 0
    for annotation_path in annotation_files:
        record = load_record(annotation_path, image_root=image_root, config=config)
        box_count += len(record.boxes)
        image_path = resolve_image_path(annotation_path, record.image_path, image_root)
        samples.extend(build_sft_samples(record, config, image_path=image_path))

    if dry_run:
        print(json.dumps(samples[:3], ensure_ascii=False, indent=2))
    else:
        write_jsonl(output_path, samples)
    return {"annotations": len(annotation_files), "boxes": box_count, "samples": len(samples)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Qwen3-VL grounding SFT data from xywh JSON annotations.")
    parser.add_argument("--config", help="JSON config path. Defaults to built-in config.")
    parser.add_argument("--dry-run", action="store_true", help="Print a preview instead of writing JSONL.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    stats = run(args.config, dry_run=args.dry_run)
    print(
        f"Processed {stats['annotations']} annotation file(s), "
        f"{stats['boxes']} box(es), generated {stats['samples']} sample(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

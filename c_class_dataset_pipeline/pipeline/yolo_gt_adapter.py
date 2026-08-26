"""适配器 A：把已有的 YOLO 格式标注（`class_id cx cy w h`，归一化坐标）
转换成 qweb3vl_grouding_vqa_lp_gai 能吃的标注结构。

用途：管道联调 / 验证。COCO128 自带 YOLO 格式的标准答案（labels/*.txt），
不需要真的去调 vlm-bbox-labeling 后面的 VLM 服务，就能把"框 -> SFT 语料"
这一段跑通、看到最终样本长什么样。

生产环境应该换成 bbox_service_adapter.py（真正调用 VLM 服务做预标注）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .deps import ClassTable, read_image_size

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_image_for_label(label_path: Path, images_dir: Path) -> Optional[Path]:
    """YOLO 目录约定：labels/xxx.txt 对应 images/xxx.jpg（后缀不定）。"""
    stem = label_path.stem
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_yolo_txt(label_path: Path, table: ClassTable) -> List[Dict[str, Any]]:
    """解析一个 YOLO 标注文件，返回 [(label, cx, cy, w, h), ...]（均为归一化坐标）。"""
    rows = []
    if not label_path.exists():
        return rows
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return rows
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        name = table.get_name(class_id)
        if name is None:
            # 类别表里没有这个编号，跳过而不是报错中断——脏数据不该拖垮整批。
            continue
        rows.append({"label": name, "cx": cx, "cy": cy, "w": w, "h": h})
    return rows


def yolo_row_to_xywh_pixel(row: Dict[str, Any], img_w: int, img_h: int) -> Dict[str, float]:
    """YOLO 的 [cx, cy, w, h]（归一化、框中心点）转换成
    qweb3vl 期望的 {x, y, width, height}（像素、左上角）。
    """
    box_w = row["w"] * img_w
    box_h = row["h"] * img_h
    x = row["cx"] * img_w - box_w / 2
    y = row["cy"] * img_h - box_h / 2
    return {"x": round(x, 2), "y": round(y, 2), "width": round(box_w, 2), "height": round(box_h, 2)}


def build_annotation_payload(
    label_path: Path,
    images_dir: Path,
    table: ClassTable,
    sample_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """单个 YOLO 标注文件 -> qweb3vl 标注 payload。没有目标框或找不到图片时返回 None。"""
    image_path = find_image_for_label(label_path, images_dir)
    if image_path is None:
        return None

    rows = parse_yolo_txt(label_path, table)
    if not rows:
        return None

    img_w, img_h = read_image_size(image_path)

    shapes = [
        {"label": row["label"], "points": yolo_row_to_xywh_pixel(row, img_w, img_h)}
        for row in rows
    ]

    return {
        "id": sample_id or label_path.stem,
        "image": str(image_path),
        "image_width": img_w,
        "image_height": img_h,
        "shapes": shapes,
    }


def iter_annotation_payloads(
    labels_dir: Path,
    images_dir: Path,
    table: ClassTable,
):
    """扫描一个 YOLO labels 目录，逐条 yield 标注 payload。"""
    for label_path in sorted(labels_dir.glob("*.txt")):
        payload = build_annotation_payload(label_path, images_dir, table)
        if payload is not None:
            yield payload

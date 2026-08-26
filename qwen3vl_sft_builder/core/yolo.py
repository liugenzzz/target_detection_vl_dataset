"""YOLO 标注解析与图片尺寸读取。

标注格式：每行 `class_id cx cy w h`，坐标为 0~1 归一化，中心点+宽高。
这是 Ultralytics / Roboflow 导出的标准格式，业务数据（jsmb_9w）就是这个格式。

图片尺寸直接读文件头，不依赖 Pillow —— 少一个部署依赖。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class Box:
    """一个标注框。cx/cy/w/h 为 0~1 归一化坐标。"""
    index: int
    class_id: int
    label: str
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area_ratio(self) -> float:
        return self.w * self.h

    def short_side_px(self, img_w: int, img_h: int) -> float:
        return min(self.w * img_w, self.h * img_h)


@dataclass
class Annotation:
    stem: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    boxes: List[Box]


# ---------------------------------------------------------------- 图片尺寸
def _png_size(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    return None


def _jpeg_size(data: bytes) -> Optional[Tuple[int, int]]:
    if not data.startswith(b"\xff\xd8"):
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        i += 2
        while marker == 0xFF and i < len(data):
            marker = data[i]
            i += 1
        if marker in {0xD8, 0xD9}:
            continue
        if i + 2 > len(data):
            return None
        seg = struct.unpack(">H", data[i:i + 2])[0]
        if seg < 2 or i + seg > len(data):
            return None
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            h, w = struct.unpack(">HH", data[i + 3:i + 7])
            return int(w), int(h)
        i += seg
    return None


def _bmp_size(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) >= 26 and data[:2] == b"BM":
        w, h = struct.unpack("<ii", data[18:26])
        return abs(int(w)), abs(int(h))
    return None


def read_image_size(path: Path) -> Tuple[int, int]:
    data = path.read_bytes()[:65536]
    size = _png_size(data) or _jpeg_size(data) or _bmp_size(data)
    if size is None:
        raise ValueError(f"无法读取图片尺寸：{path}")
    return size


# ---------------------------------------------------------------- 标注解析
def find_image(label_path: Path, images_dir: Path) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{label_path.stem}{ext}"
        if p.exists():
            return p
    return None


def parse_label_file(label_path: Path, table) -> List[Box]:
    """解析一个 YOLO 标注文件。类别表里没有的 class_id 直接跳过 ——
    脏数据不该拖垮整批构建。"""
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    boxes: List[Box] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        label = table.get_name(class_id)
        if label is None:
            continue
        boxes.append(Box(len(boxes), class_id, label, cx, cy, w, h))
    return boxes


def iter_annotations(labels_dir: Path, images_dir: Path, table,
                     sanity_max_boxes: int = 1000) -> Iterator[Annotation]:
    """遍历标注目录，逐条 yield Annotation。找不到图片或无有效框的跳过。"""
    for label_path in sorted(Path(labels_dir).glob("*.txt")):
        image_path = find_image(label_path, Path(images_dir))
        if image_path is None:
            continue
        boxes = parse_label_file(label_path, table)
        if not boxes or len(boxes) > sanity_max_boxes:
            continue
        try:
            w, h = read_image_size(image_path)
        except (ValueError, OSError):
            continue
        yield Annotation(label_path.stem, image_path, label_path, w, h, boxes)

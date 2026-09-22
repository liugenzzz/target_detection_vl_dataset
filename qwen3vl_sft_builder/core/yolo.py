"""YOLO 标注解析与图片尺寸读取。

标注格式：每行 `class_id cx cy w h`，坐标为 0~1 归一化，中心点+宽高。
这是 Ultralytics / Roboflow 导出的标准格式，业务数据（jsmb_9w）就是这个格式。

图片尺寸直接读文件头，不依赖 Pillow —— 少一个部署依赖。
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

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
    """单个查找。批量遍历请用 index_images —— 见那里的说明。"""
    for ext in IMAGE_EXTS:
        p = images_dir / f"{label_path.stem}{ext}"
        if p.exists():
            return p
    return None


def index_images(images_dir: Path) -> Dict[str, Path]:
    """{主名: 图片路径}，只列一次目录。

    【为什么不逐个 exists()】原来每个标注都要按 5 个后缀逐个 stat，11 万个
    标注就是最多 55 万次 stat。网络挂载上一次几毫秒，合计十几分钟，而且
    全程不吭声，和卡死长得一模一样。列一次目录只是一次操作。

    对完全找不到图的那批更亏：fitrs 那 4,851 个标注每个都要白试满 5 次。
    """
    out: Dict[str, Path] = {}
    for p in images_dir.iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            out.setdefault(p.stem, p)
    return out


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


# 清单文件里存图片名的字段，按这个顺序找第一个能用的。
# 各家选片流程吐出来的字段名不统一，与其要求对方改格式，不如都认。
#
# 【顺序有讲究】selected_image / label_path 排在最前：选片流程通常会把选中的
# 图另存一份并重新命名，这两个字段指的是【落盘之后】的文件，而 image /
# source 之类往往还指着原始数据集里的路径。
#
# 【不认 id】曾经把 id 放进兜底，结果取到的是上游源数据集的编号
# （"P0002_patch_000040"），和落盘文件名（"aerial_98813e8..."）毫无关系，
# 却又长得像个主名，于是静默匹配到 0 条。宁可不认，也不要认错。
MANIFEST_KEYS = ("selected_image", "label_path", "image_path", "image",
                 "images", "file_name", "filename", "img", "img_path",
                 "path", "selection_id", "stem")

# 清单里逐图声明的任务禁令 / 许可。
MANIFEST_DISALLOW_KEYS = ("disallowed_tasks", "disallow_tasks", "deny_tasks")
MANIFEST_PERMIT_KEYS = ("permitted_tasks", "permit_tasks", "allow_tasks")


@dataclass(frozen=True)
class ManifestEntry:
    """清单里关于一张图的信息。

    disallow / permit 存的是【原样】的条目，可能带后缀限定词，
    例如 "detect_class_without_completeness_audit" —— 匹配见 blocks()。
    """
    stem: str
    disallow: frozenset = frozenset()
    permit: frozenset = frozenset()

    def blocks(self, task: str) -> bool:
        """这张图上该不该禁掉这个任务（只看 disallow 那一路）。

        条目可能写成 "<任务名>_<限定词>"（真实见过
        "detect_class_without_completeness_audit"），所以前缀也算命中 ——
        上游在说「没做完整性审计之前别出 detect_class」，那就是别出。
        """
        return any(d == task or d.startswith(task + "_") for d in self.disallow)


def _first_str(row: dict, keys) -> Optional[str]:
    for key in keys:
        v = row.get(key)
        if isinstance(v, list):
            v = v[0] if v else None
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _str_set(row: dict, keys) -> frozenset:
    for key in keys:
        v = row.get(key)
        if isinstance(v, (list, tuple)):
            return frozenset(str(x).strip() for x in v if str(x).strip())
    return frozenset()


def load_manifest(path: Path | str) -> Dict[str, ManifestEntry]:
    """读选片清单，返回 {图片主名: ManifestEntry}。

    每行可以是一个 JSON 对象（从 MANIFEST_KEYS 里找图片名），也可以直接是
    一行文件名 —— 两种都认，省得为了格式再写一个转换脚本。

    【匹配用主名】清单里写绝对路径、相对路径还是裸文件名都行，一律取
    basename 去后缀。标注文件名和图片主名一致，所以这个键两边都对得上。

    【逐图任务禁令】清单里的 disallowed_tasks 会被原样带出来。选片流程知道
    一些我们看不出来的事，最典型的是「这张图的标注不保证穷尽」——
    那时「图中有多少辆车」「框出所有的车」「有没有 X」全都答不对，因为标注
    可能漏了目标。而我们这边的 all_kept 只发现得了【被自己过滤掉】的框，
    发现不了标注员根本没画的。这种错样本看不出错，只能靠上游明说。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到选片清单：{path}")
    out: Dict[str, ManifestEntry] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = _first_str(row, MANIFEST_KEYS)
            if not value:
                continue
            stem = Path(value).stem
            out[stem] = ManifestEntry(stem,
                                      _str_set(row, MANIFEST_DISALLOW_KEYS),
                                      _str_set(row, MANIFEST_PERMIT_KEYS))
        else:
            stem = Path(line).stem
            out[stem] = ManifestEntry(stem)
    return out


def iter_annotations(labels_dir: Path, images_dir: Path, table,
                     sanity_max_boxes: int = 1000,
                     keep_stems: Optional[Set[str]] = None) -> Iterator[Annotation]:
    """遍历标注目录，逐条 yield Annotation。找不到图片或无有效框的跳过。

    keep_stems 非空时只处理主名在其中的那些 —— 上游选过片，这里就只跑选中的
    那批，不必把图单独拷一个目录出来。
    """
    index = index_images(Path(images_dir))
    for label_path in sorted(Path(labels_dir).glob("*.txt")):
        if keep_stems is not None and label_path.stem not in keep_stems:
            continue
        image_path = index.get(label_path.stem)
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

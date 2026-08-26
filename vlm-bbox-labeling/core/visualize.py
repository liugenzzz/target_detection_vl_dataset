"""在原图上画框，生成人工肉眼复核用的验证图。"""

import base64
import io
import logging
import os
from typing import List

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# 容器里常见的中文字体路径，找不到就退回默认字体（中文会显示成方块）
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# 正常框的配色：刻意不含红色系 —— 红色专门留给问题框，
# 否则"轮到红色的正常框"和"被判无效的问题框"肉眼分不出来。
_COLORS = [
    (52, 199, 89), (0, 122, 255), (255, 149, 0), (175, 82, 222),
    (90, 200, 250), (255, 204, 0), (0, 199, 190), (162, 132, 94),
]
_PROBLEM_COLOR = (255, 0, 0)


def _get_font(size: int = 16):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_detections(image_bytes: bytes, detections: List[dict], draw_invalid: bool = True) -> str:
    """画框并返回 base64 编码的 JPEG。

    颜色约定：
      彩色细框            = 正常框，会写进 YOLO 标注文件
      粗红框 + [!] 前缀   = 被判为无效或需复核，不会写进标注文件

    draw_invalid=False 时不画问题框，用来看"最终真正进训练集的是什么样"。
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(max(14, img.width // 60))

    for idx, det in enumerate(detections):
        bbox = det.get("bbox_pixel")
        if not bbox:
            continue

        # 注意：判断依据必须是 valid，不能是 issues。
        # issues 只是"需要留意"，有 issue 的框仍然是有效框、仍然会写进标注文件；
        # 如果按 issues 跳过，一旦出现给所有框统一打标记的告警，验证图会变成一张空图。
        if not det.get("valid") and not draw_invalid:
            continue

        has_issue = bool(det.get("issues")) or not det.get("valid")

        x1, y1, x2, y2 = bbox
        color = _PROBLEM_COLOR if has_issue else _COLORS[idx % len(_COLORS)]
        width = 4 if has_issue else 2

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        label = f"{det.get('class_id', '?')} {det.get('class_name', '')}"
        conf = det.get("confidence")
        if conf is not None:
            label += f" ({conf})"
        if has_issue:
            # 用纯 ASCII 标记，避免字体缺字形显示成方块
            label = "[!] " + label

        # 标签底色，避免文字和图像内容糊在一起看不清
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = len(label) * 8, 16
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), label, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_base64_image(b64: str, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))

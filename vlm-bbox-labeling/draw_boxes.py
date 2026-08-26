"""把模型返回的坐标画回图片上，肉眼确认框对不对齐。

用法：

  # 方式1：把模型输出的 JSON 存成文件（可以带 ```json 包裹，脚本会自动处理）
  python draw_boxes.py 000000000544.jpg boxes.json

  # 方式2：不给 json 文件，脚本会提示你直接粘贴模型输出，粘完按两次回车
  python draw_boxes.py 000000000544.jpg

坐标格式默认按 0~1000 千分比处理（实测 Qwen3.6 就是这种）。
如果换了模型返回像素坐标，加 --mode pixel。

依赖：pip install pillow
"""

import argparse
import json
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

COLORS = [
    (255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 149, 0),
    (175, 82, 222), (255, 45, 85), (0, 199, 190), (255, 204, 0),
]


def get_font(size):
    # Windows 用黑体，Linux 用文泉驿，都找不到就用默认字体
    for p in [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def extract_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("没找到 JSON 数组")
    return json.loads(text[start : end + 1])


def read_boxes(json_path):
    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            return extract_json(f.read())

    print("请粘贴模型返回的 JSON（粘完后按回车，再按 Ctrl+Z 回车 / Linux 是 Ctrl+D）：")
    return extract_json(sys.stdin.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("json_file", nargs="?", default=None)
    ap.add_argument("--mode", default="per_mille",
                    choices=["per_mille", "pixel", "relative"],
                    help="模型返回的坐标格式，默认千分比")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    W, H = img.size
    print(f"图片尺寸：{W} x {H}")

    boxes = read_boxes(args.json_file)
    print(f"读到 {len(boxes)} 个框，坐标模式：{args.mode}\n")

    draw = ImageDraw.Draw(img)
    font = get_font(max(14, W // 45))

    for i, b in enumerate(boxes):
        raw = b.get("bbox_2d") or b.get("bbox") or b.get("box")
        if not raw or len(raw) != 4:
            print(f"  [{i}] 跳过：没有可用坐标 {b}")
            continue

        x1, y1, x2, y2 = [float(v) for v in raw]
        if args.mode == "per_mille":
            x1, x2 = x1 / 1000 * W, x2 / 1000 * W
            y1, y2 = y1 / 1000 * H, y2 / 1000 * H
        elif args.mode == "relative":
            x1, x2 = x1 * W, x2 * W
            y1, y2 = y1 * H, y2 * H

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        ratio = (x2 - x1) * (y2 - y1) / (W * H)
        suspicious = ratio >= 0.5
        color = (255, 0, 0) if suspicious else COLORS[i % len(COLORS)]

        label = str(b.get("label") or b.get("class_name") or b.get("class_id") or "")
        mark = " [!]" if suspicious else ""
        print(f"  [{i}] {label:10} 原始{raw} -> 像素[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]  占图{ratio:.1%}{mark}")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4 if suspicious else 2)

        text = f"{i}:{label}" + ("[!]" if suspicious else "")
        try:
            tb = draw.textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = len(text) * 9, 16
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), text, fill=(255, 255, 255), font=font)

    out = args.output or (os.path.splitext(args.image)[0] + "_verify.jpg")
    img.save(out, quality=92)
    print(f"\n已保存：{out}")
    print("打开看一眼：框套在物体上 = 坐标模式对了；框整体偏移或缩在角落 = 模式选错了，换 --mode 再试。")


if __name__ == "__main__":
    main()

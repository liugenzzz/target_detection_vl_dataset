#!/usr/bin/env python
"""下载 VisDrone2019-DET 验证集并转成 YOLO 格式，用来复现本项目的测试结果。

为什么用 VisDrone 而不是 COCO 做主要测试集：它是无人机航拍、小目标极密集
（548 图 / 38759 框，每图均 70 个目标，短边中位数只有 18px），和业务数据的
航拍视角、小目标、人员密集三个特征都对得上。COCO 每图才 7 个目标，
测不出密集场景下的问题 —— 本项目「每图框数上限」那条规则就是错误设计，
在 COCO 上完全看不出来，换到 VisDrone 才暴露出会跳过 88% 的图。

它也有覆盖不到的地方：全是日间彩色图，测不了夜视/红外场景。

    python scripts/get_visdrone.py --out ./data/visdrone

跑完按提示把三个路径填进 config/local.yaml 即可。
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.yolo import read_image_size  # noqa: E402

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"

# VisDrone 标注每行：x,y,w,h,score,category,truncation,occlusion
# category: 0=ignored 11=others 都要丢弃；1..10 映射到 YOLO 的 0..9
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
         "tricycle", "awning-tricycle", "bus", "motor"]


def download(dest: Path) -> Path:
    zip_path = dest / "VisDrone2019-DET-val.zip"
    if zip_path.exists():
        print(f"已存在，跳过下载：{zip_path}")
        return zip_path
    dest.mkdir(parents=True, exist_ok=True)
    print(f"下载中（约 78MB）：{URL}")

    def hook(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(URL, zip_path, reporthook=hook)
    print()
    return zip_path


def convert(root: Path, out: Path) -> tuple[int, int, int]:
    ann_dir, img_dir = root / "annotations", root / "images"
    labels_out = out / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)

    n_img = n_kept = n_dropped = 0
    for ann in sorted(ann_dir.glob("*.txt")):
        img = img_dir / f"{ann.stem}.jpg"
        if not img.exists():
            continue
        W, H = read_image_size(img)
        n_img += 1
        lines = []
        for line in ann.read_text().strip().splitlines():
            parts = [p for p in line.replace(" ", "").split(",") if p != ""]
            if len(parts) < 6:
                continue
            x, y, w, h, _score, cat = (int(parts[i]) for i in range(6))
            if cat in (0, 11) or w <= 0 or h <= 0:
                n_dropped += 1
                continue
            lines.append(f"{cat - 1} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                         f"{w / W:.6f} {h / H:.6f}")
            n_kept += 1
        (labels_out / f"{ann.stem}.txt").write_text("\n".join(lines))

    (out / "classes.yaml").write_text(
        f"nc: {len(NAMES)}\nnames:\n" + "\n".join(f'  {i}: "{n}"' for i, n in enumerate(NAMES)),
        encoding="utf-8")

    link = out / "images"
    if not link.exists():
        try:
            link.symlink_to(img_dir.resolve(), target_is_directory=True)
        except OSError:                       # Windows 无权限时退回复制
            import shutil
            shutil.copytree(img_dir, link)
    return n_img, n_kept, n_dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("./data/visdrone"))
    args = ap.parse_args()
    out = args.out.resolve()

    zip_path = download(out)
    root = out / "VisDrone2019-DET-val"
    if not root.exists():
        print("解压中…")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)

    n_img, n_kept, n_dropped = convert(root, out)
    print(f"\n转换完成：{n_img} 张图，{n_kept} 个框"
          f"（丢弃 ignored/others {n_dropped} 个）")
    print("\n把这三行填进 config/local.yaml：\n")
    print("paths:")
    print(f'  labels_dir:   "{out / "labels"}"')
    print(f'  images_dir:   "{out / "images"}"')
    print(f'  classes_yaml: "{out / "classes.yaml"}"')
    print("\n然后：python scripts/analyze.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

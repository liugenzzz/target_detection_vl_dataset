"""批量跑图脚本 —— 100 张图手动调接口太痛苦，用这个。

用法（服务已经起来的前提下）：

    python batch_run.py --input ./images

输出目录结构：
    results/
      labels/     xxx.txt          YOLO 标注文件，可直接用于训练
      verify/     xxx_verify.jpg   画好框的验证图（人工复核看这个）
      raw/        xxx.json         完整返回，含模型原始输出
      summary.csv                  每张图汇总，Excel 可直接打开
"""

import argparse
import base64
import csv
import json
import os
import sys
import time

import requests

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(input_dir):
    return [
        os.path.join(input_dir, n)
        for n in sorted(os.listdir(input_dir))
        if os.path.splitext(n)[1].lower() in IMAGE_EXTS
    ]


def run_one(base_url, image_path, timeout):
    url = f"{base_url.rstrip('/')}/api/v1/detect"
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        resp = requests.post(url, files=files, data={"draw": "true"}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def save_result(out_dir, image_path, result):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for sub in ("labels", "verify", "raw"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    with open(os.path.join(out_dir, "labels", f"{stem}.txt"), "w", encoding="utf-8") as f:
        f.write(result.get("yolo_txt", ""))

    b64 = result.get("annotated_image_base64")
    if b64:
        with open(os.path.join(out_dir, "verify", f"{stem}_verify.jpg"), "wb") as f:
            f.write(base64.b64decode(b64))

    with open(os.path.join(out_dir, "raw", f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="图片目录")
    ap.add_argument("--output", default="./results", help="结果输出目录")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    images = find_images(args.input)
    if not images:
        print(f"目录 {args.input} 下没有找到图片")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    print(f"共 {len(images)} 张图，输出到 {args.output}")

    rows = []
    t0 = time.time()
    for i, path in enumerate(images, 1):
        name = os.path.basename(path)
        try:
            result = run_one(args.base_url, path, args.timeout)
            save_result(args.output, path, result)
            s = result.get("stats", {})
            rows.append({
                "image": name,
                "detections": s.get("total", 0),
                "valid": s.get("valid", 0),
                "invalid": s.get("invalid", 0),
                "flagged": s.get("flagged_for_review", 0),
                "distinct_classes": s.get("distinct_classes", 0),
                "elapsed_sec": result.get("elapsed_sec"),
                "error": result.get("error", ""),
            })
            print(f"{i}/{len(images)} {name} -> {s.get('valid', 0)} 个有效框，{result.get('elapsed_sec')}s")
        except Exception as e:
            print(f"{i}/{len(images)} {name} -> 失败：{e}")
            rows.append({"image": name, "detections": 0, "valid": 0, "invalid": 0,
                         "flagged": 0, "distinct_classes": 0, "elapsed_sec": None, "error": str(e)})

    csv_path = os.path.join(args.output, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total_valid = sum(r["valid"] for r in rows)
    failed = sum(1 for r in rows if r["error"])
    print(f"\n完成：{len(images)} 张图，共 {total_valid} 个有效框，{failed} 张失败，"
          f"总耗时 {round(time.time() - t0, 1)}s")
    print(f"结果目录：{args.output}")


if __name__ == "__main__":
    main()

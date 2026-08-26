"""适配器 B：调用真正跑起来的 vlm-bbox-labeling 服务，对图片做 VLM 自动预标注，
再把它的返回结构转换成 qweb3vl_grouding_vqa_lp_gai 能吃的标注 payload。

生产环境（专业领域图片、347 类业务类别）应该走这条路径，而不是
yolo_gt_adapter.py（那个只用于拿 COCO128 这类"自带标准答案"的开源数据跑通管道）。

依赖 vlm-bbox-labeling 已经按其 README 起好服务：
    docker compose up -d --build
    curl http://localhost:8000/health
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class BBoxServiceError(RuntimeError):
    pass


def detect_image(
    image_path: Path,
    base_url: str = "http://localhost:8000",
    timeout: int = 300,
) -> Dict[str, Any]:
    """调用 POST /api/v1/detect，返回原始 JSON。"""
    url = f"{base_url.rstrip('/')}/api/v1/detect"
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        resp = requests.post(url, files=files, data={"draw": "false"}, timeout=timeout)
    if resp.status_code != 200:
        raise BBoxServiceError(f"{image_path}: 检测失败 HTTP {resp.status_code} {resp.text[:300]}")
    return resp.json()


def detection_result_to_annotation_payload(
    result: Dict[str, Any],
    image_path: Path,
    sample_id: Optional[str] = None,
    only_valid: bool = True,
) -> Optional[Dict[str, Any]]:
    """vlm-bbox-labeling 的 /api/v1/detect 返回结构 -> qweb3vl 标注 payload。

    只取 `valid: true` 的框（对应 README 里说的"直接存成 .txt 就是标注文件"那批），
    `issues` 非空但仍标记为有效的框会带着人工复核标记一起进来，
    彻底无效（valid=false）的框默认丢弃，避免脏框污染训练语料。
    """
    image_info = result.get("image", {})
    img_w = int(image_info.get("width") or 0)
    img_h = int(image_info.get("height") or 0)
    if img_w <= 0 or img_h <= 0:
        return None

    shapes: List[Dict[str, Any]] = []
    for det in result.get("detections", []):
        if only_valid and not det.get("valid"):
            continue
        label = det.get("class_name")
        bbox = det.get("bbox_pixel")
        if not label or not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        shapes.append(
            {
                "label": str(label),
                "points": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                "flags": {"needs_review": bool(det.get("issues"))},
            }
        )

    if not shapes:
        return None

    return {
        "id": sample_id or image_path.stem,
        "image": str(image_path),
        "image_width": img_w,
        "image_height": img_h,
        "shapes": shapes,
    }


def iter_annotation_payloads_via_service(
    images_dir: Path,
    base_url: str = "http://localhost:8000",
    timeout: int = 300,
    image_exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp"),
):
    """扫描一个图片目录，逐张调用 VLM 预标注服务，yield 标注 payload。

    调用量大时建议直接用 vlm-bbox-labeling/batch_run.py 先把 raw/*.json 落盘，
    再复用 detection_result_to_annotation_payload() 离线转换，避免服务偶发抖动
    导致整个大批量任务从头重跑。
    """
    for name in sorted(os.listdir(images_dir)):
        path = images_dir / name
        if path.suffix.lower() not in image_exts:
            continue
        result = detect_image(path, base_url=base_url, timeout=timeout)
        payload = detection_result_to_annotation_payload(result, path)
        if payload is not None:
            yield payload

"""跨任务一致性核对：同一张图，八个任务说出来的目标数量必须对得上。

一张图会被抽出多条样本、分给不同任务。它们各自独立生成，但说的是同一张图：

    inventory_locate  「图中有 3 名人员、2 辆卡车。」
    detect_class      「框出图中所有的人员。」-> 必须正好 3 个框
    exist_negative    「图中有没有人员？」-> 有，图中有 3 名人员
    exist_negative    「图中有没有直升机？」-> 没有  -> 别的样本里就不能出现直升机

任何一条对不上，就是同一张图配了两套真值，模型只能学成随机猜。

结构上这已经由「一张图只有一份 kept 集合喂给全部任务」保证了，但保证和
验证过是两回事 —— 曾经因为 clean_labels 按任务重算过一次可用类别，
同一张图上一个样本说没有人员、另一个又去定位人员。这个模块把话说死：
每次构建都核对一遍，对不上就报出来。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

_BOX_JSON = re.compile(r'\{.*"bbox_2d".*\}|\[\s*\{.*"bbox_2d".*\}\s*\]', re.DOTALL)


def _boxes_in(answer: str) -> List[Tuple[str, Tuple[int, ...]]]:
    """从一条答案里抠出 (label, bbox) 列表。不是框答案就返回空。"""
    text = answer.strip()
    if not text.startswith(("{", "[")):
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    items = data if isinstance(data, list) else [data]
    out = []
    for it in items:
        if isinstance(it, dict) and "bbox_2d" in it:
            out.append((str(it.get("label", "")), tuple(it["bbox_2d"])))
    return out


def claims_of(sample: Dict[str, Any]) -> Dict[str, Any]:
    """把一条样本对「这张图上有什么」的主张提取出来。"""
    meta = sample.get("metadata") or {}
    task = meta.get("task_type", "")
    convs = sample.get("conversations") or []
    answers = [t["value"] for t in convs if t.get("from") == "gpt"]

    boxes: Dict[str, set] = defaultdict(set)
    for a in answers:
        for label, bbox in _boxes_in(a):
            boxes[label].add(bbox)

    counts: Dict[str, int] = {}
    absent: List[str] = []
    if task == "inventory_locate":
        # metadata.inventory 形如 ["人员x3", "卡车x2"]
        for item in meta.get("inventory") or []:
            label, _, n = str(item).rpartition("x")
            if label and n.isdigit():
                counts[label] = int(n)
    elif task == "detect_class" and meta.get("n_boxes"):
        counts[meta["label"]] = int(meta["n_boxes"])
    elif task == "exist_negative":
        if meta.get("polarity") == "negative":
            absent.append(meta["label"])

    return {"task": task, "boxes": dict(boxes), "counts": counts, "absent": absent}


def check(samples: List[Dict[str, Any]], kept_labels: Dict[str, Dict[str, int]]
          ) -> Dict[str, Any]:
    """核对全部样本。kept_labels: {图片路径: {类别: 过滤后框数}}。

    返回 {"checked": 核对的图片数, "violations": [...]}。
    """
    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        images = s.get("images") or []
        if images:
            by_image[images[0]].append(s)

    violations: List[str] = []
    for image, group in by_image.items():
        truth = kept_labels.get(image, {})
        stated: Dict[str, int] = {}
        said_absent: set = set()
        seen_labels: set = set()

        for s in group:
            c = claims_of(s)
            task = c["task"]
            seen_labels |= set(c["boxes"])
            said_absent |= set(c["absent"])

            for label, n in c["counts"].items():
                if label in stated and stated[label] != n:
                    violations.append(
                        f"{image}：{label} 的数量说法不一致（{stated[label]} vs {n}，"
                        f"后者来自 {task}）")
                stated[label] = n
                if truth and truth.get(label, n) != n:
                    violations.append(
                        f"{image}：{task} 说有 {n} 个 {label}，"
                        f"但过滤后实际是 {truth.get(label, 0)} 个")

            for label, bs in c["boxes"].items():
                if truth and len(bs) > truth.get(label, 0):
                    violations.append(
                        f"{image}：{task} 给出 {len(bs)} 个 {label} 的框，"
                        f"超过过滤后的 {truth.get(label, 0)} 个")

        for label in said_absent & seen_labels:
            violations.append(f"{image}：一条样本答「没有{label}」，另一条却框出了 {label}")
        for label in said_absent:
            if truth.get(label):
                violations.append(f"{image}：答「没有{label}」，但过滤后还有 "
                                  f"{truth[label]} 个 {label}")

    return {"checked_images": len(by_image), "violations": violations}

"""解析模型输出的 JSON。

模型偶尔会加 ```json 包裹、加前后说明文字、或者输出被截断，
这里做多层容错，尽量把能救的结果救回来，而不是整张图直接判失败。
"""

import json
import logging
import re
from typing import List

logger = logging.getLogger(__name__)


def extract_json(text: str):
    """从模型输出里提取 JSON 对象或数组。解析不出来时抛 ValueError。"""
    if not text or not text.strip():
        raise ValueError("模型输出为空")

    s = text.strip()

    # 1. 去掉 markdown 代码块包裹
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()

    # 2. 直接尝试解析
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 3. 截取第一个 [ 或 { 到最后一个 ] 或 } 之间的内容
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = s.find(open_ch)
        end = s.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                continue

    # 4. 输出被截断的情况：逐个抠出完整的 {...} 对象，丢掉最后残缺的那个
    salvaged = _salvage_objects(s)
    if salvaged:
        logger.warning("JSON 解析失败，已抢救出 %d 条完整记录（可能有截断丢失）", len(salvaged))
        return salvaged

    raise ValueError(f"无法从模型输出中解析出 JSON。原始输出前500字符：{text[:500]}")


def _salvage_objects(s: str) -> List[dict]:
    """从残缺文本里逐个抠出配对完整的 JSON 对象。

    要能处理嵌套结构：模型返回 {"bbox_format":..., "objects":[{...},{...}]} 时，
    如果输出被截断，最外层的 { 永远等不到闭合。所以不能只在 depth 归零时收集，
    必须用栈记录每一层的起点，任意一层闭合就尝试解析 —— 否则内层那些完整的框会全部丢失。
    """
    out = []
    stack = []
    in_str = False
    escape = False

    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue

        if ch == "{":
            stack.append(i)
        elif ch == "}":
            if not stack:
                continue
            start = stack.pop()
            try:
                obj = json.loads(s[start : i + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)

    # 只保留看起来像检测结果的对象（含坐标字段），滤掉外层包装
    box_keys = ("bbox_2d", "bbox", "box", "bounding_box", "coordinates", "坐标", "x1")
    boxes = [o for o in out if any(k in o for k in box_keys)]
    return boxes if boxes else out


def as_list(parsed) -> List[dict]:
    """把解析结果统一成 list[dict]。

    模型有时会包一层，比如 {"objects": [...]} 或 {"detections": [...]}。
    """
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        for key in ("objects", "detections", "results", "data", "items", "boxes"):
            if key in parsed and isinstance(parsed[key], list):
                return [x for x in parsed[key] if isinstance(x, dict)]
        # 只有一个键、且值是「由对象组成的list」的情况才当作外层包装。
        # 注意不能把 bbox 这种数字数组误判成包装层。
        vals = [
            v for v in parsed.values()
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
        ]
        if len(vals) == 1:
            return vals[0]
        # 本身就是单条记录
        return [parsed]
    return []


def as_id_list(parsed) -> List[int]:
    """粗筛阶段：把解析结果转成编号列表。

    兼容 [1,2,3] / [{"class_id":1}, ...] / {"class_ids":[1,2]} 等多种写法。
    """
    ids = []

    def _push(v):
        try:
            ids.append(int(str(v).strip()))
        except (ValueError, TypeError):
            pass

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                for key in ("class_id", "id", "编号", "index"):
                    if key in item:
                        _push(item[key])
                        break
            else:
                _push(item)
    elif isinstance(parsed, dict):
        for key in ("class_ids", "ids", "classes", "objects", "results", "编号"):
            if key in parsed and isinstance(parsed[key], list):
                return as_id_list(parsed[key])

    # 去重保序
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out

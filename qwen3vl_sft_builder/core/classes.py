"""类别表加载，以及编号<->名称的双向校验。

思路移植自 vlm-bbox-labeling/core/classes.py（本项目不 import 它，保持独立部署）。

额外增加了「易混类别组」检测：业务类别表里存在名称包含关系的类别，例如
    人员 / 一般人员 / 军事人员
    剪刀 / PVC管剪刀 / 修枝剪
    切管器 / 切管机
这些类别靠视觉难以可靠区分。构建时不阻塞（标注文件已经指定了 class_id，
直接查表取名即可），但会在样本 metadata 里打标记，训练后若这几类混淆严重，
可以据此快速定位到是哪批样本。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)


class ClassTable:
    def __init__(self, id2name: Dict[int, str]):
        self.id2name = id2name
        self.name2id: Dict[str, int] = {}
        for cid, name in id2name.items():
            key = self._norm(name)
            if key in self.name2id:
                logger.warning("类别表存在重名：'%s'（编号 %s 和 %s）", name, self.name2id[key], cid)
            self.name2id[key] = cid
        self._confusable, self._hypernym = _detect_confusable(id2name)

    @staticmethod
    def _norm(name: str) -> str:
        return str(name).strip().lower().replace(" ", "").replace("　", "")

    @property
    def count(self) -> int:
        return len(self.id2name)

    def get_name(self, class_id: int) -> Optional[str]:
        return self.id2name.get(class_id)

    def get_id(self, name: str) -> Optional[int]:
        return self.name2id.get(self._norm(name))

    def is_confusable(self, class_id: int) -> bool:
        """该类别是否属于某个易混组。"""
        return class_id in self._confusable

    def hypernym_group(self, class_id: int) -> List[str]:
        """与该类别存在【上下位关系】的类别名（名字互相包含）。

        这一组不能拿来做拒答样本：图里有遮阳三轮车，问「有没有三轮车」
        答「没有」是错的 —— 遮阳三轮车本来就是三轮车。
        """
        group = self._hypernym.get(class_id)
        return sorted(group) if group else []

    def confusable_group(self, class_id: int) -> List[str]:
        """返回与该类别互相易混的类别名列表（含自身）。不属于任何组则返回 []。"""
        group = self._confusable.get(class_id)
        return sorted(group) if group else []

    def confusable_summary(self) -> Dict[str, List[str]]:
        """全部易混组，用于构建报告。{代表类别名: [组内全部类别名]}"""
        seen: Dict[str, List[str]] = {}
        for cid, group in self._confusable.items():
            key = min(group)
            if key not in seen:
                seen[key] = sorted(group)
        return seen


def _detect_confusable(id2name: Dict[int, str]):
    """检测名称相近的类别组。返回 (易混组, 上下位关系组)。

    两条判据：
      1. 包含关系：'人员' 是 '一般人员' / '军事人员' 的子串 -> 三者互为易混。
      2. 等长且只差一个字：'切管器' vs '切管机'、'压接钳' vs '压管钳' ——
         这类一字之差的类别最容易混，靠视觉几乎不可能可靠区分。
    只在长度 >= 2 的名称之间判断，避免把 'suv' 之类的短名误判。
    """
    names = {cid: str(n).strip() for cid, n in id2name.items()}
    groups: Dict[int, Set[str]] = {}
    hypernym: Dict[int, Set[str]] = {}
    items = [(cid, n) for cid, n in names.items() if len(n) >= 2]

    for cid_a, name_a in items:
        for cid_b, name_b in items:
            if cid_a >= cid_b:
                continue
            contained = name_a in name_b or name_b in name_a
            if contained or _one_char_apart(name_a, name_b):
                groups.setdefault(cid_a, {name_a}).add(name_b)
                groups.setdefault(cid_b, {name_b}).add(name_a)
            if contained:
                # 包含关系多半是上下位词（三轮车 ⊂ 遮阳三轮车、人员 ⊂ 军事人员）。
                # 标记出来，因为它对【拒答样本】是有害的：图里有遮阳三轮车，
                # 问「有没有三轮车」答「没有」是错的 —— 遮阳三轮车本来就是三轮车。
                # 一字之差的（切管器 vs 切管机）是并列的不同东西，答「没有」才成立。
                hypernym.setdefault(cid_a, set()).add(name_b)
                hypernym.setdefault(cid_b, set()).add(name_a)
    return groups, hypernym


def _one_char_apart(a: str, b: str) -> bool:
    """等长且恰好只有一个字符不同。"""
    if len(a) != len(b) or a == b:
        return False
    return sum(1 for x, y in zip(a, b) if x != y) == 1


def load_class_table(path: str | Path) -> ClassTable:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到类别表文件：{path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names")
    if names is None:
        raise ValueError(f"{path} 中没有 names 字段")

    if isinstance(names, dict):
        id2name = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        id2name = {i: str(v) for i, v in enumerate(names)}
    else:
        raise ValueError(f"names 字段格式无法识别：{type(names)}")

    declared = data.get("nc")
    if declared is not None and int(declared) != len(id2name):
        logger.warning("yaml 里 nc=%s，实际解析出 %s 个类别", declared, len(id2name))

    table = ClassTable(id2name)
    logger.info("已加载类别表 %s，共 %d 个类别，其中 %d 个属于易混组",
                path, table.count, sum(1 for c in id2name if table.is_confusable(c)))
    return table

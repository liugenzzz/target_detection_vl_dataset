"""加载 YOLO 格式的类别表 yaml，并提供编号<->名称的双向校验。

支持两种常见写法：

    nc: 347
    names:
      0: "N95防护口罩"
      1: "PVC管剪刀"
      ...

或者列表写法：

    names: ["N95防护口罩", "PVC管剪刀", ...]
"""

import logging
import os
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ClassTable:
    def __init__(self, id2name: Dict[int, str]):
        self.id2name: Dict[int, str] = id2name
        # 名称 -> 编号。注意：如果类别表里有重名，后面的会覆盖前面的，这里会告警
        self.name2id: Dict[str, int] = {}
        for cid, name in id2name.items():
            key = self._norm(name)
            if key in self.name2id:
                logger.warning("类别表存在重名：'%s'（编号 %s 和 %s）", name, self.name2id[key], cid)
            self.name2id[key] = cid

    @staticmethod
    def _norm(name: str) -> str:
        """名称归一化：去空格、统一大小写，容忍模型输出时的细微差异。"""
        return str(name).strip().lower().replace(" ", "").replace("　", "")

    @property
    def count(self) -> int:
        return len(self.id2name)

    def get_name(self, class_id: int) -> Optional[str]:
        return self.id2name.get(class_id)

    def get_id(self, name: str) -> Optional[int]:
        return self.name2id.get(self._norm(name))

    def format_full_list(self) -> str:
        """拼成给模型看的完整类别清单，用于粗筛阶段的 prompt。"""
        lines = [f"{cid}: {name}" for cid, name in sorted(self.id2name.items())]
        return "\n".join(lines)

    def format_subset(self, class_ids) -> str:
        """只拼指定编号的类别清单，用于精标阶段的 prompt。"""
        lines = []
        for cid in class_ids:
            name = self.id2name.get(cid)
            if name is not None:
                lines.append(f"{cid}: {name}")
        return "\n".join(lines)

    def validate(self, class_id, class_name):
        """校验模型给的编号和名称是否自洽。

        返回 (归一化后的class_id, 归一化后的class_name, issues列表)。
        issues 非空表示这条结果需要人工重点复核。
        """
        issues = []

        # 编号可能被模型输出成字符串 "214" 或 "0214"
        cid = None
        if class_id is not None:
            try:
                cid = int(str(class_id).strip())
            except (ValueError, TypeError):
                issues.append(f"class_id 无法解析为整数：{class_id!r}")

        name_id = self.get_id(class_name) if class_name else None

        if cid is not None and cid not in self.id2name:
            issues.append(f"class_id {cid} 不在类别表中")
            cid = None

        if class_name and name_id is None:
            issues.append(f"class_name '{class_name}' 不在类别表中")

        # 核心的双重校验：编号和名称互相印证
        if cid is not None and name_id is not None and cid != name_id:
            issues.append(
                f"编号与名称不一致：编号{cid}对应'{self.id2name[cid]}'，"
                f"但名称'{class_name}'对应编号{name_id}"
            )
            # 冲突时以名称为准（模型对文字的把握通常好于对数字编号的把握），
            # 但一定会打上 issue 标记，人工复核时能筛出来
            cid = name_id

        # 只给了其中一个的情况，用类别表补全另一个
        if cid is None and name_id is not None:
            cid = name_id
        final_name = self.id2name.get(cid) if cid is not None else class_name

        return cid, final_name, issues


_cached_table: Optional[ClassTable] = None


def load_class_table(path: str, force_reload: bool = False) -> ClassTable:
    global _cached_table
    if _cached_table is not None and not force_reload:
        return _cached_table

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到类别表文件：{path}\n"
            "请确认 CLASSES_YAML_PATH 配置正确，且该文件已挂载进容器。"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if names is None:
        raise ValueError(f"{path} 中没有找到 names 字段")

    if isinstance(names, dict):
        id2name = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        id2name = {i: str(v) for i, v in enumerate(names)}
    else:
        raise ValueError(f"names 字段格式无法识别：{type(names)}")

    declared_nc = data.get("nc")
    if declared_nc is not None and int(declared_nc) != len(id2name):
        logger.warning("yaml 里 nc=%s，但实际解析出 %s 个类别，请检查类别表", declared_nc, len(id2name))

    logger.info("已加载类别表 %s，共 %d 个类别", path, len(id2name))
    _cached_table = ClassTable(id2name)
    return _cached_table

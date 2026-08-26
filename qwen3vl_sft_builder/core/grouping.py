"""从标注/图片文件名推导「原始来源分组」，用于 train/val 划分。

为什么需要这个：实测业务数据的文件名形如

    Wasserfalle-...-bei-Hochwasser_mp4-14_jpg.rf.062732212bcc96d202df7b978c5e2987.txt
    Wasserfalle-...-bei-Hochwasser_mp4-15_jpg.rf.b1db71c78b1af7ab43b8e41758441100.txt
    y2mate_com-Creciente-Subita-del-Rio-Cesar_480p_mp4-8_jpg.rf.88ec9a5c58aca819852502af31322842.txt
    y2mate_com-Creciente-Subita-del-Rio-Cesar_480p_mp4-8_jpg.rf.b97e1a6cca79ec5cef2146cbdff6bf90.txt

暴露出两层重复：

  1. `.rf.<hash>` 是 Roboflow 导出后缀。**同一张原图**做不同数据增强会导出成
     多个文件，只有 hash 不同（上面 mp4-8 那两行就是）。
  2. `_mp4-<帧号>` 说明图片是**视频抽帧**。同一视频的相邻帧画面几乎一样
     （mp4-14 / -15 / -16）。

所以按「图片」随机划分 train/val 是不够的 —— 同一张原图的两个增强版本、
或同一视频的相邻帧，会被拆到训练集和验证集两边，验证集里全是训练时见过的画面，
指标虚高且看不出来。

正确做法：按本模块推导出的 group_key 分组，**同一组整组进 train 或整组进 val**。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List

# Roboflow 导出后缀：<原图名>.rf.<32位hex>
_ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{6,}$", re.IGNORECASE)

# 视频抽帧后缀：<视频名>_mp4-<帧号> / -mp4-<帧号>，也兼容 avi/mov/mkv
_VIDEO_FRAME = re.compile(
    r"[_-](?:mp4|avi|mov|mkv|flv|wmv)[_-]\d+$", re.IGNORECASE
)

# 常见的「图片扩展名被并进文件名」的写法：xxx_jpg / xxx_png
_EMBEDDED_EXT = re.compile(r"_(?:jpg|jpeg|png|bmp|webp)$", re.IGNORECASE)


def source_group_key(filename: str) -> str:
    """把一个标注/图片文件名归约成它的『原始来源』标识。

    同一视频的所有帧、同一原图的所有增强版本，都会归约到同一个 key。

    >>> source_group_key("clip_mp4-14_jpg.rf.062732212bcc96d202df7b978c5e2987.txt")
    'clip'
    >>> source_group_key("clip_mp4-15_jpg.rf.b1db71c78b1af7ab43b8e41758441100.txt")
    'clip'
    >>> source_group_key("voc8_9948.txt")
    'voc8_9948'
    """
    stem = Path(str(filename)).stem

    # .rf.<hash> 去掉后，Path.stem 可能只剥掉了一层，循环剥干净
    prev = None
    while prev != stem:
        prev = stem
        stem = _ROBOFLOW_SUFFIX.sub("", stem)
        stem = Path(stem).stem if stem.endswith((".jpg", ".png", ".jpeg")) else stem

    stem = _EMBEDDED_EXT.sub("", stem)
    stem = _VIDEO_FRAME.sub("", stem)
    return stem or Path(str(filename)).stem


def group_files(filenames: Iterable[str]) -> Dict[str, List[str]]:
    """按来源分组。返回 {group_key: [文件名, ...]}。"""
    groups: Dict[str, List[str]] = {}
    for name in filenames:
        groups.setdefault(source_group_key(name), []).append(name)
    return groups


def split_by_group(
    filenames: Iterable[str],
    val_ratio: float = 0.05,
    seed: int = 20260826,
):
    """按来源分组划分 train/val，保证同一组不跨越两边。

    返回 (train_files, val_files)。用固定种子保证可复现。
    """
    import random

    groups = group_files(filenames)
    keys = sorted(groups)  # 先排序，再用固定种子打乱 —— 保证与输入顺序无关
    random.Random(seed).shuffle(keys)

    target_val = len(list(filenames)) * val_ratio if not isinstance(filenames, list) else len(filenames) * val_ratio
    val_files: List[str] = []
    train_files: List[str] = []
    for key in keys:
        if len(val_files) < target_val:
            val_files.extend(groups[key])
        else:
            train_files.extend(groups[key])
    return sorted(train_files), sorted(val_files)

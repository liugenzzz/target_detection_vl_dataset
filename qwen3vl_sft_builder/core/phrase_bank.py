"""扩充问法库的读写与校验。

prompts/ 下每个「多问法」文件手写只有五六条。十万条样本摊下来，同一句问话要
出现上千次 —— 模型学到的会是「见到这句口令就输出框」，而不是「听懂要框什么」。
这里把每个池子扩到几十条，同一句的复现率降一个量级。

问法与图片内容无关（「那辆车在哪？」跟图片长什么样没关系），所以跟量词表一样
一次性生成、长期复用，不进构建时的调用预算。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml

import prompts

# 生成的句子里允许出现的前缀垃圾：序号、项目符号、引号
_JUNK = re.compile(r'^\s*(?:[-*·•]|\d+[.、)）]|\(\d+\)|["“”\'‘’])+\s*')
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def sanitize(line: str) -> str:
    """去掉模型爱加的序号、项目符号和包裹引号。"""
    line = _JUNK.sub("", line.strip())
    return line.strip().strip('"“”\'‘’').strip()


def visible_len(line: str) -> int:
    """句子长度。占位符按两个字算 —— {label} 实到值是「三轮车」这种短词，
    按字面 7 个字算会把本来合格的短句误判成超长。"""
    return len(_PLACEHOLDER.sub("字字", line))


def accept(name: str, line: str, required: Sequence[str], max_len: int,
           seen: Iterable[str]) -> bool:
    """一条生成结果是否收得下。不合格的直接丢，不做修补 ——
    问法池是要进十万条训练数据的，宁可少几条也不能混进坏句子。"""
    if not line or line.startswith("#"):
        return False
    if visible_len(line) > max_len:
        return False
    if set(_PLACEHOLDER.findall(line)) != set(required):
        # 占位符对不上：少了会让问句失去指向（「那辆在哪？」），
        # 多了会在 .format() 时抛 KeyError 把整批构建打断。
        return False
    try:
        line.format(**{k: "占位" for k in required})
    except (KeyError, IndexError, ValueError):
        # 单个大括号、格式说明符等，format 会炸
        return False
    return line not in set(seen)


def load(path: Path | str | None) -> Dict[str, List[str]]:
    """读问法库。文件不存在返回空 dict —— 没生成过也能正常构建。"""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    banks = data.get("phrase_banks") if isinstance(data, dict) else None
    if not isinstance(banks, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for name, phrases in banks.items():
        if isinstance(phrases, list):
            out[str(name)] = [str(x) for x in phrases]
    return out


def dump(banks: Dict[str, List[str]]) -> str:
    """写成 yaml。这个文件的主要读者是人 —— 生成完要一眼扫过去把别扭的句子删掉，
    所以带上说明注释，中文不转义，key 排序固定，方便 diff。"""
    header = "\n".join([
        "# 扩充问法库，由 scripts/build_phrase_banks.py 生成。",
        "# 每个池子对应 prompts/<名字>.txt，构建时与该文件里手写的几条合并取样。",
        "# 觉得哪句别扭，直接把那行删掉即可，不用重新生成。",
        "",
    ])
    body = yaml.safe_dump({"phrase_banks": {k: list(banks[k]) for k in sorted(banks)}},
                          allow_unicode=True, sort_keys=True,
                          default_flow_style=False, width=10 ** 6)
    return header + body


def install(cfg) -> int:
    """按 config 把问法库装进 prompts。返回装载的说法总条数。"""
    banks = load(cfg.get_path("phrase_banks.path", ""))
    prompts.use_bank(banks)
    return sum(len(v) for v in banks.values())

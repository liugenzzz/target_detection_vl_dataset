"""扩充问法库的读写与校验。

prompts/ 下每个「多问法」文件手写只有五六条。十万条样本摊下来，同一句问话要
出现上千次 —— 模型学到的会是「见到这句口令就输出框」，而不是「听懂要框什么」。
这里把每个池子扩到几十条，同一句的复现率降一个量级。

问法与图片内容无关（「那辆车在哪？」跟图片长什么样没关系），所以跟量词表一样
一次性生成、长期复用，不进构建时的调用预算。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml

import prompts

from . import register

logger = logging.getLogger(__name__)

# 生成的句子里允许出现的前缀垃圾：序号、项目符号、引号
_JUNK = re.compile(r'^\s*(?:[-*·•]|\d+[.、)）]|\(\d+\)|["“”\'‘’])+\s*')
_PLACEHOLDER = re.compile(r"\{(\w+)\}")
# 问句必须以句号、问号或叹号收尾。模型爱在正文外多写一句「以下是几种写法：」，
# 剥掉序号后它结构上完全合法（无占位符、无禁用词、长度也够短），
# 会一路混进问法库当成问句用 —— 实测出现过「这条带了序号」当第二轮问句。
_TERMINATED = re.compile(r"[。？！?!]$")
# 模型的元话语：它在讲「我要写几种说法」，而不是在写那句说法本身
_META = ("以下", "如下", "例如", "比如", "示例", "写法", "说法", "第一句",
         "这条", "这句", "上面", "下面", "注意")


def sanitize(line: str) -> str:
    """去掉模型爱加的序号、项目符号和包裹引号。"""
    line = _JUNK.sub("", line.strip())
    return line.strip().strip('"“”\'‘’').strip()


def visible_len(line: str) -> int:
    """句子长度。占位符按两个字算 —— {label} 实到值是「三轮车」这种短词，
    按字面 7 个字算会把本来合格的短句误判成超长。"""
    return len(_PLACEHOLDER.sub("字字", line))


def accept(name: str, line: str, required: Sequence[str], max_len: int,
           seen: Iterable[str], forbidden: Sequence[str] = (),
           optional_refer: bool = False,
           require_any: Sequence[str] = ()) -> bool:
    """一条生成结果是否收得下。不合格的直接丢，不做修补 ——
    问法池是要进十万条训练数据的，宁可少几条也不能混进坏句子。

    optional_refer：这一轮的指代能由上文承接（「它周围是什么情况？」紧跟在
    刚给出框的那一轮后面，不带类别名也说得通），此时允许整句不带占位符。
    但**不允许只带一半** —— 只剩「这{mw}」或量词对不上的「这三轮车」，
    比干脆不提还糟。
    """
    if not line or line.startswith("#"):
        return False
    if register.problems(line):
        # 句法闸：句末缺标点、或者是模型的元话语（「以下是几种写法：」）。
        # 剥掉序号后这些结构上完全合法，只能靠这道闸拦。
        return False
    if require_any and not any(w in line for w in require_any):
        # 语义漏了：detect_class 问的是穷举，少了「所有」就和 ground_unique 撞车
        return False
    if any(w in line for w in forbidden):
        # 语义跑偏：往「描述」池里塞了「在哪」。结构上完全合法，只能靠禁用词拦。
        return False
    if visible_len(line) > max_len:
        return False
    found = set(_PLACEHOLDER.findall(line))
    ok = {frozenset(required)} | ({frozenset()} if optional_refer else set())
    if frozenset(found) not in ok:
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


def install(cfg) -> Dict[str, int]:
    """按 config 把问法库装进 prompts，装之前重新校验一遍。

    这个文件是给人手改的（生成完要扫一眼删掉别扭的句子），也可能是旧版本
    留下的 —— 手一抖删掉半个占位符，构建时 .format() 直接抛异常打断整批；
    往「描述」池里粘一句「在哪」，则会静默产出问非所答的样本。
    所以入库前按当前 .txt 的占位符与禁用词再过一遍，不合格的丢掉并报出来。
    """
    banks = load(cfg.get_path("phrase_banks.path", ""))
    max_len = int(cfg.get_path("phrase_banks.max_len", 30))
    glob = tuple(cfg.get_path("phrase_banks.forbid_global", []) or [])
    kept: Dict[str, List[str]] = {}
    dropped: Dict[str, int] = {}
    for name, phrases in banks.items():
        try:
            required = prompts.placeholders_of(name)
            forbidden = prompts.forbidden_of(name) + glob
            optional = prompts.has_flag(name, "optional-refer")
            req_any = prompts.required_any_of(name)
        except FileNotFoundError:
            # 池子对应的 .txt 已经删了（任务下线），整组跳过
            dropped[name] = len(phrases)
            continue
        good: List[str] = []
        for line in phrases:
            line = sanitize(line)
            if accept(name, line, required, max_len, good, forbidden,
                      optional, req_any):
                good.append(line)
        if good:
            kept[name] = good
        if len(good) < len(phrases):
            dropped[name] = len(phrases) - len(good)
    prompts.use_bank(kept)
    if dropped:
        logger.warning("问法库有 %d 条不合格已丢弃：%s。"
                       "多半是手改时动坏了占位符，或句子跑到别的池子的意思上去了。",
                       sum(dropped.values()),
                       "、".join(f"{k} {v} 条" for k, v in sorted(dropped.items())))
    return {"loaded": sum(len(v) for v in kept.values()),
            "dropped": sum(dropped.values())}

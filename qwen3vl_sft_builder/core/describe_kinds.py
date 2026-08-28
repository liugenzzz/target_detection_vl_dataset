"""描述的【子类型】—— 按内容维度拆分，不是按措辞拆分。

原先所有描述都是一个信息结构：外观 + 方位 + 周边。给问句换再多说法，
答案的结构还是那一个，模型学到的也还是那一个 —— 换汤不换药。

真正的多样性要来自【答案里装的是什么】。这也是 Osprey 的做法：它的区域描述
覆盖 object category / type / action / location / color / status / attributes
这些内容维度，靠 PACO-LVIS 的 456 个部件类和 55 种属性撑起多样性，
不是靠换问法。

七种子类型各写一个提示词文件（prompts/describe/*.txt），可以单独改：

    appearance  只说物体本身，一个字不提方位和周边
    state       只说状态和动作
    part        聚焦一个部位展开
    position    只说方位
    relation    只说和周围什么挨着
    contrast    和图中同类比有什么不同
    full        综合三段式（保留，但降到少数）

【代码指定 + 模型可拒绝】：代码给每个目标派一种，模型看图觉得不适合
（小目标看不清部件、图里没有同类可比）就返回空，代码换下一种。
让模型自己挑会坍缩到最省力的那种 —— 这个我们已经见过一次了。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import prompts

PROMPT_SUBDIR = "describe"


@dataclass(frozen=True)
class Kind:
    name: str            # appearance / state / ...
    needs: str           # 适用条件，写进提示词让模型据此判断能不能做
    must_not: Tuple[str, ...]   # 答案里不该出现的词，代码侧兜底校验
    answer_spec: str     # 答案该装什么
    q_example: str
    a_example: str


def _directive(text: str, key: str) -> str:
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("#!") and f"{key}:" in ln:
            return ln.split(f"{key}:", 1)[1].strip()
    return ""


def _field(text: str, key: str) -> str:
    """取 `key: 值` 段，值可以跨行（续行以空格缩进）。"""
    out, collecting = [], False
    for ln in text.splitlines():
        if ln.startswith(f"{key}:"):
            out.append(ln.split(":", 1)[1].strip())
            collecting = True
        elif collecting:
            if ln.startswith(" ") or ln.startswith("\t"):
                out.append(ln.strip())
            else:
                break
    return " ".join(out).strip()


def load_all() -> Dict[str, Kind]:
    """读 prompts/describe/ 下的全部子类型。文件即配置，加一种就多一种。"""
    out: Dict[str, Kind] = {}
    for path in sorted((prompts.PROMPT_DIR / PROMPT_SUBDIR).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        name = _directive(text, "kind") or path.stem
        out[name] = Kind(
            name=name,
            needs=_directive(text, "needs"),
            must_not=tuple(_directive(text, "must-not").split()),
            answer_spec=_field(text, "answer-spec"),
            q_example=_field(text, "q-example"),
            a_example=_field(text, "a-example"),
        )
    if not out:
        raise ValueError(f"{prompts.PROMPT_DIR / PROMPT_SUBDIR} 下没有任何子类型文件")
    return out


def render_assignment(kind: Kind, slot: int) -> str:
    """拼给模型看的一条指派。

    按【挑中的顺序】指派，不按框号 —— 挑哪些框是模型自己定的，
    代码提前按框号指派必然对不上（实测因此 300 张图只出了 5 条样本）。
    """
    return (f"  第 {slot} 个挑中的目标 -> 「{kind.name}」\n"
            f"    适用条件：{kind.needs}\n"
            f"    答案要求：{kind.answer_spec}\n"
            f"    问句示例：{kind.q_example}\n"
            f"    答案示例：{kind.a_example}")


def answer_violates(kind: Kind, answer: str) -> str:
    """答案有没有跑出这个子类型的范围。返回命中的词，空串表示合格。

    模型很容易把「只说外观」写成「位于画面左侧的一辆红色三轮车」——
    加了方位就又滑回三段式了。这道闸是子类型能不能立住的关键：
    没有它，七种子类型跑几轮就会退化成同一种。
    """
    hit = [w for w in kind.must_not if w in answer]
    return "/".join(hit)

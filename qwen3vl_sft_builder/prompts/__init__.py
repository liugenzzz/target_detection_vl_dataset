"""提示词加载。所有提示词都是 prompts/ 下的纯文本文件，与代码分离。

改提示词不需要动代码，也不需要重装依赖 —— 服务器上直接改 .txt 即可。

目录按【任务】分：

    prompts/ground_unique/       只服务这一个任务的提示词
    prompts/inventory_locate/
    prompts/exist_negative/
    ...                          八个任务各一个目录
    prompts/_shared/             多个任务共用的（主线末轮的描述问法、短答案后缀）
    prompts/_vlm/                调 VLM 那两次用的（挑对象、描述起手方式）
    prompts/_tools/              一次性脚本用的（量词表、扩充问法库、质检）

要改某个任务的问法，进它自己的目录改就行，不会误伤别的任务。
下划线开头的三个目录是共用件，改动会影响多个任务 —— 目录名上就标出来了。

加载按【文件名】而不是路径，所以移动目录不用改任何调用点；
文件名全局唯一，重名会在加载时立刻报错而不是静默取到其中一个。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Sequence, Tuple

PROMPT_DIR = Path(__file__).resolve().parent

# 扩充问法库：{池名: (说法, ...)}。由 scripts/build_phrase_banks.py 一次性生成，
# 构建时由 use_bank() 灌进来。空的时候一切照旧，只用 .txt 里手写的那几句。
_BANK: Dict[str, Tuple[str, ...]] = {}


@lru_cache(maxsize=None)
def _index() -> Dict[str, Path]:
    """文件名 -> 路径。重名立刻报错 —— 静默取到其中一个，改了另一个却不生效，
    是最难查的那种问题。"""
    found: Dict[str, Path] = {}
    for path in sorted(PROMPT_DIR.rglob("*.txt")):
        if path.stem in found:
            raise ValueError(
                f"提示词重名：{path} 与 {found[path.stem]} 都叫 {path.stem}.txt。"
                f"加载按文件名进行，重名会取到哪一个是不确定的，请改名。")
        found[path.stem] = path
    return found


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """读取一个提示词模板。name 不带 .txt 后缀，也不带目录 —— 按文件名查找，
    所以在 prompts/ 下怎么分目录都不影响调用点。"""
    path = _index().get(name)
    if path is None:
        raise FileNotFoundError(
            f"找不到提示词 {name}.txt。当前可用：{', '.join(sorted(_index()))}")
    return path.read_text(encoding="utf-8").strip()


def path_of(name: str) -> Path:
    """提示词的实际路径，用于给用户指路（「改 prompts/xxx/yyy.txt」）。"""
    return _index()[name]


@lru_cache(maxsize=None)
def load_variants(name: str) -> tuple:
    """读一个「多问法」文件：每个非空、非注释行是一种问法。

    有些任务的问法与图像内容无关（「定位图中所有的人员」跟图片长什么样没关系），
    这类不必调 VLM 生成，用模板池随机取一句即可，零成本。
    只有指代内容依赖图像的任务（要看见才知道是什么颜色）才需要 VLM。
    """
    lines = [ln.strip() for ln in load(name).splitlines()]
    variants = tuple(ln for ln in lines if ln and not ln.startswith("#"))
    if not variants:
        raise ValueError(f"{name}.txt 里没有任何问法")
    return variants


def comment_of(name: str) -> str:
    """取一个问法文件开头的 # 注释块，去掉 # 后拼成一段话。

    每个问法池的注释都写明了这个池子在对话里干什么用（「第一轮：问图中有什么」），
    生成扩充问法时要把这段话给模型，否则它不知道该保持什么意思不变。
    """
    out = []
    for ln in load(name).splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if not ln.startswith("#"):
            break
        out.append(ln.lstrip("#").strip())
    return " ".join(out)


def forbidden_of(name: str) -> Tuple[str, ...]:
    """取一个问法池的禁用词，写在文件里形如：

        #! forbid: 在哪 位置 坐标 框

    占位符校验只管结构，管不了语义 —— 「描述一下这辆车」和「这辆车在哪」
    占位符完全一样，混进去主线第二轮就问非所答，且构建报告里看不出来。
    禁用词是这道语义闸，既用来过滤生成结果，也会写进生成提示词里先说清楚。
    """
    words: list = []
    for ln in load(name).splitlines():
        ln = ln.strip()
        if ln.startswith("#!") and "forbid:" in ln:
            words += ln.split("forbid:", 1)[1].split()
    return tuple(dict.fromkeys(words))


def optional_group_of(name: str) -> Tuple[str, ...]:
    """取一个问法池的「可整组省略」的占位符，写在文件里形如：

        #! optional-group: mw label

    这组占位符要么全在、要么全不在。全不在是合法的 —— 上一轮刚点过这个目标，
    「它周围是什么情况？」不带类别名也说得通。只带一半不行：剩下「说说这{mw}。」
    或者量词丢了的「说说这三轮车。」，比干脆不提还糟。
    """
    words: list = []
    for ln in load(name).splitlines():
        ln = ln.strip()
        if ln.startswith("#!") and "optional-group:" in ln:
            words += ln.split("optional-group:", 1)[1].split()
    return tuple(dict.fromkeys(words))


def required_any_of(name: str) -> Tuple[str, ...]:
    """取一个问法池的必含词（至少命中一个），写在文件里形如：

        #! require-any: 所有 全部 每一个

    禁用词管「不能是别的意思」，必含词管「必须是这个意思」。detect_class
    问的是穷举（「框出图中所有的人员」），少了「所有」就和 ground_unique
    撞车 —— 同一个问句配两种答案，模型只能学成随机猜要给一个还是给全部。
    """
    words: list = []
    for ln in load(name).splitlines():
        ln = ln.strip()
        if ln.startswith("#!") and "require-any:" in ln:
            words += ln.split("require-any:", 1)[1].split()
    return tuple(dict.fromkeys(words))


def placeholders_of(name: str) -> Tuple[str, ...]:
    """一个问法池用到的占位符集合，例如 inv_ask_box 是 (label, mw)。"""
    found = set()
    for v in load_variants(name):
        found.update(re.findall(r"\{(\w+)\}", v))
    return tuple(sorted(found))


def use_bank(bank: Dict[str, Sequence[str]]) -> None:
    """装载扩充问法库。可重复调用，后一次覆盖前一次。"""
    _BANK.clear()
    for name, phrases in (bank or {}).items():
        cleaned = tuple(dict.fromkeys(str(p).strip() for p in phrases if str(p).strip()))
        if cleaned:
            _BANK[name] = cleaned


def variants(name: str) -> Tuple[str, ...]:
    """该问法池实际可用的全部说法 = 手写的 + 扩充库里的，按顺序去重。

    手写的那几句是校准过的基准，不因为有了扩充库就丢掉；两边合并取样，
    问法总数从个位数涨到几十条，同一句话在整个数据集里的复现率随之降一个量级。
    """
    return tuple(dict.fromkeys(load_variants(name) + _BANK.get(name, ())))


def render_choice(name: str, rng, **kwargs) -> str:
    """从多问法池里随机取一句并填充占位符。"""
    return rng.choice(variants(name)).format(**kwargs)


SPLIT = " ||| "


def pick_pair(name: str, rng) -> Tuple[str, str]:
    """从一个「说明 ||| 示例」的池子里随机取一条，返回 (说明, 示例)。

    提示词里给固定的例子，模型会朝那几个例子的句式收敛 —— 例子越少收敛得越死。
    每次调用换一条要求和示例，等于把「例子」这个变量本身也随机化了。
    """
    line = rng.choice(load_variants(name))
    if SPLIT not in line:
        raise ValueError(f"{name}.txt 的每一行都要写成「说明{SPLIT}示例」，"
                         f"这一行没有分隔符：{line}")
    rule, example = line.split(SPLIT, 1)
    return rule.strip(), example.strip()


def render(name: str, **kwargs) -> str:
    """读取并填充占位符。缺占位符会明确报错，不静默出错。"""
    template = load(name)
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        raise KeyError(f"提示词 {name}.txt 需要占位符 {exc}，但调用时没有提供") from exc

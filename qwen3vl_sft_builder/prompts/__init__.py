"""提示词加载。所有提示词都是 prompts/ 下的纯文本文件，与代码分离。

改提示词不需要动代码，也不需要重装依赖 —— 服务器上直接改 .txt 即可。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """读取一个提示词模板。name 不带 .txt 后缀。"""
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in PROMPT_DIR.glob("*.txt")))
        raise FileNotFoundError(f"找不到提示词 {path}。当前可用：{available}")
    return path.read_text(encoding="utf-8").strip()


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


def render_choice(name: str, rng, **kwargs) -> str:
    """从多问法文件里随机取一句并填充占位符。"""
    return rng.choice(load_variants(name)).format(**kwargs)


def render(name: str, **kwargs) -> str:
    """读取并填充占位符。缺占位符会明确报错，不静默出错。"""
    template = load(name)
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        raise KeyError(f"提示词 {name}.txt 需要占位符 {exc}，但调用时没有提供") from exc

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


def render(name: str, **kwargs) -> str:
    """读取并填充占位符。缺占位符会明确报错，不静默出错。"""
    template = load(name)
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        raise KeyError(f"提示词 {name}.txt 需要占位符 {exc}，但调用时没有提供") from exc

#!/usr/bin/env python
"""把 prompts/ 下的「多问法」池子从手写的五六条扩到几十条，一次性跑。

    python scripts/build_phrase_banks.py                 # 全部池子
    python scripts/build_phrase_banks.py --pool inv_ask_what --show
    python scripts/build_phrase_banks.py --target 60 --force

问法跟图片内容无关，所以这是纯文本调用，跟量词表一样生成一次长期复用，
不占构建时的调用预算。全部池子加起来大约十几次调用。

生成结果写到 config/phrase_banks.yaml，构建时自动与 .txt 里手写的几条合并。
**生成完请扫一遍那个文件**，别扭的句子直接删行，删完不用重新生成。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prompts                                   # noqa: E402
from config import load_config                   # noqa: E402
from core import phrase_bank                     # noqa: E402
from core.vlm_client import VlmClient            # noqa: E402

# 一个池子最多问几轮。够 target 就停；一直凑不够也不能无限问下去 ——
# 有些池子（「图中有什么？」）本身可说的花样就有限，问到后面全是重复。
MAX_ROUNDS = 6


def _placeholder_rule(required) -> str:
    if not required:
        return "- 句子里不要出现任何大括号。\n"
    shown = "、".join("{" + k + "}" for k in required)
    return (f"- 每句都必须原样包含 {shown} 这些占位符，一个不能少、一个不能多，"
            f"大括号和里面的英文都不要改。\n"
            f"  它们在真正使用时会被替换成具体的词，你只管把它们摆在通顺的位置上。\n")


def _forbid_rule(forbidden) -> str:
    if not forbidden:
        return ""
    return ("- 这几个词一个都不能出现："
            + "、".join(forbidden)
            + "。\n  它们属于别的问法池，混进来会让这一轮问非所答。\n")


def expand(client: VlmClient, name: str, target: int, batch: int,
           max_len: int, temperature: float, verbose: bool) -> list[str]:
    """扩充一个池子，返回新增的说法（不含 .txt 里手写的）。"""
    seeds = list(prompts.load_variants(name))
    required = prompts.placeholders_of(name)
    forbidden = prompts.forbidden_of(name)
    purpose = prompts.comment_of(name) or f"图片问答对话里的一句话（{name}）"
    seen = list(seeds)
    fresh: list[str] = []
    rejected = 0

    for rnd in range(MAX_ROUNDS):
        if len(seen) >= target:
            break
        want = min(batch, target - len(seen))
        text = prompts.render(
            "gen_phrases",
            purpose=purpose,
            seeds="\n".join(seen[-30:]),      # 只给最近的，太长模型会开始抄前面
            n=want + 5,                       # 多要几条，抵掉被丢掉的
            placeholder_rule=_placeholder_rule(required),
            forbid_rule=_forbid_rule(forbidden),
        )
        raw = client._post({
            "model": client.model,
            "messages": [{"role": "user", "content": text}],
            "temperature": temperature,
            "max_tokens": 1024,
        })
        if not raw:
            print(f"  [{name}] 第 {rnd + 1} 轮调用失败")
            continue
        got = 0
        for line in raw.splitlines():
            line = phrase_bank.sanitize(line)
            if phrase_bank.accept(name, line, required, max_len, seen, forbidden):
                seen.append(line)
                fresh.append(line)
                got += 1
            elif line:
                rejected += 1
        if verbose:
            print(f"  [{name}] 第 {rnd + 1} 轮：收 {got} 条，累计 {len(seen)}")
        if got == 0:
            break        # 一条都收不到，再问也是白问

    print(f"  {name:18s} 手写 {len(seeds):2d} + 新增 {len(fresh):2d} = "
          f"{len(seeds) + len(fresh):2d} 条（丢弃 {rejected}）")
    return fresh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--pool", action="append", help="只扩充指定池子，可重复")
    ap.add_argument("--target", type=int, help="覆盖 phrase_banks.target")
    ap.add_argument("--force", action="store_true", help="已有的池子也重新生成")
    ap.add_argument("--show", action="store_true", help="打印每个池子的最终全部说法")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = args.out or Path(cfg.get_path("phrase_banks.path", "config/phrase_banks.yaml"))
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    target = args.target or int(cfg.get_path("phrase_banks.target", 40))
    batch = int(cfg.get_path("phrase_banks.batch", 20))
    max_len = int(cfg.get_path("phrase_banks.max_len", 30))
    temperature = float(cfg.get_path("phrase_banks.temperature", 1.0))
    pools = args.pool or list(cfg.get_path("phrase_banks.pools", []) or [])
    if not pools:
        raise SystemExit("phrase_banks.pools 是空的，没有要扩充的池子")

    client = VlmClient(cfg)
    if not client.enabled:
        raise SystemExit("vlm.enabled 为 false，无法生成问法库")

    banks = phrase_bank.load(out)
    print(f"共 {len(pools)} 个池子，目标每个 {target} 条：")
    for name in pools:
        if banks.get(name) and not args.force:
            print(f"  {name:18s} 已有 {len(banks[name])} 条，跳过（--force 重新生成）")
            continue
        banks[name] = expand(client, name, target, batch, max_len, temperature,
                             verbose=args.show)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(phrase_bank.dump(banks), encoding="utf-8")
    print(f"\n已写入 {out}")

    if args.show:
        prompts.use_bank(banks)
        for name in pools:
            print(f"\n===== {name} =====")
            for v in prompts.variants(name):
                print("  " + v)

    print("\n请扫一遍上面的文件，别扭的句子直接删掉那一行 —— 删完不用重新生成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

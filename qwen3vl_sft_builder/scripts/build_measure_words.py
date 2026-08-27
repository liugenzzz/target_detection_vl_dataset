#!/usr/bin/env python
"""为类别表生成量词表，一次性跑，结果缓存到 config/measure_words.yaml。

量词只跟类别有关、跟图片无关（船是「艘」、车是「辆」、人是「名」），
所以不必每张图去问，347 个类别问一次就够了 —— 这是纯文本调用，不发图片。

    python scripts/build_measure_words.py

生成后 config 里的 quality.measure_words_path 会自动读取它。
没有这个文件时全部退回「个」，不影响构建，只是量词不地道。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prompts                                   # noqa: E402
from config import load_config                   # noqa: E402
from core.classes import load_class_table        # noqa: E402
from core.vlm_client import VlmClient            # noqa: E402

BATCH = 60          # 一次问 60 个，太多模型会漏答


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config" / "measure_words.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    table = load_class_table(cfg.require("paths.classes_yaml"))
    client = VlmClient(cfg)
    if not client.enabled:
        raise SystemExit("vlm.enabled 为 false，无法生成量词表")

    names = sorted(set(table.id2name.values()))
    result: dict[str, str] = {}
    for i in range(0, len(names), BATCH):
        chunk = names[i:i + BATCH]
        prompt = prompts.render("measure_words", names="、".join(chunk))
        raw = client._post({                       # 纯文本，不带图片
            "model": client.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 2048,
        })
        if not raw:
            print(f"  第 {i // BATCH + 1} 批失败，这批退回「个」")
            continue
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            continue
        try:
            got = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        for k, v in got.items():
            v = str(v).strip()
            if k in chunk and len(v) == 1:
                result[k] = v
        print(f"  第 {i // BATCH + 1} 批：{len(chunk)} 个词 -> 拿到 {len(result)} 个量词")

    missing = [n for n in names if n not in result]
    lines = ["# 类别量词表，由 scripts/build_measure_words.py 生成。",
             "# 量词只跟类别有关、跟图片无关，所以一次性生成、长期复用。",
             "# 未覆盖的类别在构建时退回「个」。", "measure_words:"]
    for n in names:
        lines.append(f'  "{n}": "{result.get(n, "个")}"')
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n共 {len(names)} 个类别，拿到 {len(result)} 个量词，"
          f"{len(missing)} 个退回「个」")
    print(f"已写入 {args.out}")
    if missing[:8]:
        print(f"  未覆盖示例：{missing[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

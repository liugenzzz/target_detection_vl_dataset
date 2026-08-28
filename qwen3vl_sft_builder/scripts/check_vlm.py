#!/usr/bin/env python
"""接入新的模型服务后，第一个该跑的脚本 —— 别跳过。

三步递进地验证，每一步失败都会告诉你该改哪里：

    1. 纯文本连通性   地址 / 端口 / model 名 / api_key 对不对
    2. 图片输入       这个部署开没开多模态，收不收 base64 图片
    3. 真实提示词     用 prompts/vlm_describe.txt 跑一次，看返回能不能解析成 JSON

用法：
    python scripts/check_vlm.py                      # 用配置里的图片目录随便挑一张
    python scripts/check_vlm.py --image 某张图.jpg
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prompts                                   # noqa: E402
from config import load_config
from core.cli import _cli  # noqa: E402
from core.vlm_client import _endpoints_from                   # noqa: E402
from core.yolo import IMAGE_EXTS                 # noqa: E402

OK, BAD = "  [通过]", "  [失败]"


def post(url, key, payload, timeout):
    import requests
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.time()
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    return r, time.time() - t0


def content_of(resp_json):
    return resp_json["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--image", help="测试用图片；不给则从 images_dir 里挑第一张")
    args = ap.parse_args()

    cfg = load_config(args.config)
    timeout = int(cfg.get_path("vlm.timeout", 300))
    endpoints = _endpoints_from(cfg.get_path("vlm", {}) or {})

    # 模型池里有几路就逐路自检 —— 只测第一路的话，第二路配错要等到跑全量
    # 才会暴露，那时已经烧掉几小时。
    if len(endpoints) > 1:
        print(f"模型池共 {len(endpoints)} 路，逐路自检：")
        for ep in endpoints:
            print(f"  {ep.name}  model={ep.model}  concurrency={ep.concurrency}")
        print("=" * 66)
    failed = []
    for ep in endpoints:
        if len(endpoints) > 1:
            print(f"\n{'=' * 66}\n检查 {ep.name}\n{'=' * 66}")
        if _check_one(ep, timeout, cfg, args) != 0:
            failed.append(ep.name)
    if failed:
        print(f"\n{BAD} 模型池里这几路没通过：{'、'.join(failed)}")
        return 1
    return 0


def _check_one(ep, timeout, cfg, args) -> int:
    url, key, model = ep.api_url, ep.api_key, ep.model

    print(f"服务地址 : {url}")
    print(f"模型名   : {model}")
    print(f"api_key  : {'已设置（' + str(len(key)) + ' 字符）' if key else '未设置'}")
    print("-" * 66)

    # ---------------------------------------------------------- 1 纯文本
    print("\n[1/3] 纯文本连通性")
    try:
        r, el = post(url, key, {"model": model, "max_tokens": 16,
                                "messages": [{"role": "user", "content": "回复OK两个字"}]}, timeout)
    except Exception as exc:                                   # noqa: BLE001
        print(f"{BAD} 连不上：{exc}")
        print("\n  排查：地址端口对不对？服务起没起？容器里能不能访问到这个地址？")
        return 1
    if r.status_code != 200:
        print(f"{BAD} HTTP {r.status_code}：{r.text[:300]}")
        print("\n  排查：401/403 -> api_key 不对；404 -> 路径不对（要带 /v1/chat/completions）；"
              "\n        400 且提示 model -> model 名和服务上的对不上")
        return 1
    try:
        print(f"{OK} {el:.2f}s，返回：{content_of(r.json())[:60]!r}")
    except (KeyError, IndexError, ValueError):
        print(f"{BAD} 返回结构不是 OpenAI 兼容格式：{r.text[:300]}")
        return 1

    # ---------------------------------------------------------- 2 图片输入
    print("\n[2/3] 图片输入（多模态）")
    if args.image:
        img = Path(args.image)
    else:
        d = Path(cfg.get_path("paths.images_dir", "") or ".")
        cand = [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS] if d.is_dir() else []
        if not cand:
            print(f"{BAD} 找不到测试图片，用 --image 指定一张")
            return 1
        img = cand[0]
    print(f"  用图：{img.name}")

    mime = "png" if img.suffix.lower() == ".png" else "jpeg"
    b64 = base64.b64encode(img.read_bytes()).decode()
    img_msg = [{"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
               {"type": "text", "text": "这张图里有什么？一句话。"}]
    try:
        r, el = post(url, key, {"model": model, "max_tokens": 128,
                                "messages": [{"role": "user", "content": img_msg}]}, timeout)
    except Exception as exc:                                   # noqa: BLE001
        print(f"{BAD} 请求异常：{exc}")
        return 1
    if r.status_code != 200:
        print(f"{BAD} HTTP {r.status_code}：{r.text[:300]}")
        print("\n  排查：多半是这个部署没开多模态输入，或者不认 image_url 这种写法。"
              "\n        换支持视觉的模型，或确认部署方式（vLLM 要带 --limit-mm-per-prompt）")
        return 1
    print(f"{OK} {el:.2f}s，返回：{content_of(r.json())[:100]!r}")
    print(f"  单次图片请求约 {el:.1f} 秒 —— 用它估算并发数和总耗时")

    # ---------------------------------------------------------- 3 真实提示词
    print("\n[3/3] 真实提示词（prompts/vlm_describe.txt）")
    prompt_text = prompts.render("vlm_describe", bbox=[300, 400, 500, 600],
                                 label="测试目标", siblings_hint="")
    msg = [{"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
           {"type": "text", "text": prompt_text}]
    r, el = post(url, key, {"model": model, "max_tokens": int(cfg.get_path("vlm.max_tokens", 1024)),
                            "temperature": float(cfg.get_path("vlm.temperature", 0.35)),
                            "messages": [{"role": "user", "content": msg}]}, timeout)
    if r.status_code != 200:
        print(f"{BAD} HTTP {r.status_code}：{r.text[:300]}")
        return 1
    raw = content_of(r.json())
    print(f"  模型原始输出：\n    {raw[:300]}")
    parsed = _parse_vlm_json(raw)
    if parsed is None:
        print(f"\n{BAD} 解析不出 JSON。构建时这类会回落模板描述。")
        print("  排查：改 prompts/vlm_describe.txt 让模型只吐 JSON —— 改提示词不用动代码。")
        print("        小模型常见问题是加解释性前后缀，或把键名写错。")
        return 1
    print(f"\n{OK} 解析成功")
    print(f"    referring   = {parsed.referring!r}")
    print(f"    description = {parsed.description!r}")
    if not parsed.referring:
        print("  注意：referring 为空，构建时该目标会回落模板空间指代。")

    print("\n" + "=" * 66)
    print("三项全通过，可以跑构建了：")
    print("    python scripts/build.py --limit 10")
    print(f"\n单次图片请求 {el:.1f} 秒。按 10 万条样本 ≈ 10.6 万次调用估算总耗时：")
    for c in (1, 4, 8, 16):
        print(f"    并发 {c:>2} 路  ->  约 {106000 * el / c / 3600:.1f} 小时")
    print("  并发数先从 4 起步，观察服务是否稳定再往上加。")
    return 0




if __name__ == "__main__":
    raise SystemExit(_cli(main))

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
from core import describe_kinds                  # noqa: E402
from core.cli import _cli  # noqa: E402
from core.vlm_client import _diagnose, _endpoints_from, _parse_scene_json  # noqa: E402
from core.yolo import IMAGE_EXTS                 # noqa: E402

OK, BAD = "  [通过]", "  [失败]"


def _real_box_list(cfg, img: Path):
    """按管道同样的格式，拼出这张图【真实标注】的目标清单。返回 (文本, 框数)。

    找不到标注时退回假标注 —— 那样只能验 JSON 格式，验不了模型挑不挑得中。
    """
    from core.classes import load_class_table
    from core.coords import yolo_to_bbox2d
    from core.yolo import iter_annotations

    labels_dir = Path(cfg.get_path("paths.labels_dir", "") or "")
    label_file = labels_dir / f"{img.stem}.txt"
    if labels_dir.is_dir() and label_file.is_file():
        try:
            table = load_class_table(cfg.require("paths.classes_yaml"))
            scale = int(cfg.get_path("coords.scale", 1000))
            origin = int(cfg.get_path("coords.origin", 0))
            for ann in iter_annotations(labels_dir, img.parent, table,
                                        int(cfg.get_path("quality.sanity_max_boxes", 1000))):
                if ann.image_path.stem != img.stem:
                    continue
                boxes = ann.boxes[:8]
                lines = [f"图中共 {len(boxes)} 个已标注目标："]
                for b in boxes:
                    xy = yolo_to_bbox2d(b.cx, b.cy, b.w, b.h,
                                        ann.width, ann.height, scale, origin)
                    lines.append(f"  [{b.index}] {b.label}  位于 {xy}")
                if boxes:
                    return "\n".join(lines), len(boxes)
        except Exception as exc:                      # noqa: BLE001
            print(f"  注意：读真实标注失败（{exc}），退回假标注")
    fake = "\n".join(f"  [{i}] 测试目标{i}  位于 [{i * 100}, 200, {i * 100 + 80}, 300]"
                      for i in range(3))
    return fake, 0


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
    ap.add_argument("--list-models", action="store_true",
                    help="只列出这个 api_key 能用哪些模型，然后退出")
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
    if args.list_models:
        for ep in endpoints:
            _list_models(ep, timeout)
        return 0

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


def _list_models(ep, timeout) -> None:
    """列出这个 key 能用哪些模型。403「no access to model X」时最需要的就是这个 ——
    不然只能靠猜模型名。"""
    url = ep.api_url.split("/chat/completions")[0].rstrip("/") + "/models"
    print(f"\n{ep.name}\n  GET {url}")
    try:
        import requests
        headers = {"Authorization": f"Bearer {ep.api_key}"} if ep.api_key else {}
        r = requests.get(url, headers=headers, timeout=timeout)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  {BAD} 连不上：{exc}")
        return
    if r.status_code != 200:
        print(f"  {BAD} HTTP {r.status_code}：{r.text[:300]}")
        return
    try:
        names = sorted(str(m.get("id", "")) for m in r.json().get("data", []))
    except (ValueError, AttributeError):
        print(f"  {BAD} 返回结构不认识：{r.text[:300]}")
        return
    if not names:
        print(f"  {BAD} 这个 key 一个模型都没有授权，找服务方开权限。")
        return
    print(f"  {OK} 可用模型 {len(names)} 个：")
    for n in names:
        mark = "  <- 当前 vlm.model" if n == ep.model else ""
        print(f"      {n}{mark}")
    if ep.model not in names:
        print(f"\n  {BAD} 配置里的 vlm.model = {ep.model!r} 不在这个列表里，")
        print("      把它改成上面某一个（改 config/local.yaml 或 $env:VLM_MODEL）。")


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
        hint = _diagnose(r.status_code, r.text)
        print("\n  " + (hint or "重试可能能过；持续失败就找服务方看日志。").replace("\n", "\n  "))
        if r.status_code in (400, 403):
            _list_models(ep, timeout)
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
    # 这一步必须验【管道真正在用的那个提示词】。以前验的是 vlm_describe.txt，
    # 而管道早已改用 vlm_select.txt —— 自检全绿、全量跑起来才发现解析不出来。
    print("\n[3/3] 真实提示词（prompts/vlm_select.txt，管道实际用的就是它）")
    # 【用这张图的真实标注】。以前拼的是「测试目标0 位于 [0,200,80,300]」这种
    # 假框：模型看图后发现图里根本没有「测试目标0」，老实返回 "picked": []，
    # 自检却把这个正确行为报成失败。用真标注，验的才是构建时真正发出去的东西。
    box_list, n_real = _real_box_list(cfg, img)
    if n_real == 0:
        print("  注意：这张图没有对应的标注文件，退回假标注 —— "
              "只能验格式，验不了模型挑不挑得中目标。用 --image 换一张有标注的。")
    # 描述子类型的指派段也要一起发过去 —— 管道就是这么拼的，少一段就等于
    # 验了一个和线上不一样的提示词。（这里曾经漏传，check_vlm 直接崩在 render。）
    kinds = list(describe_kinds.load_all().values())
    kind_assignments = "\n\n".join(
        describe_kinds.render_assignment(kinds[i % len(kinds)], i + 1)
        for i in range(min(max(n_real, 1), 3)))
    max_pick = min(max(n_real, 1), 3)
    prompt_text = prompts.render("vlm_select", box_list=box_list, max_pick=max_pick,
                                 kind_assignments=kind_assignments)
    msg = [{"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
           {"type": "text", "text": prompt_text}]
    r, el = post(url, key, {"model": model,
                            "max_tokens": int(cfg.get_path("vlm.max_tokens", 1024)),
                            "temperature": float(cfg.get_path("vlm.temperature_select", 0.85)),
                            "messages": [{"role": "user", "content": msg}]}, timeout)
    if r.status_code != 200:
        print(f"{BAD} HTTP {r.status_code}：{r.text[:300]}")
        return 1
    raw = content_of(r.json())
    print(f"  模型原始输出：\n    {raw[:400]}")
    parsed = _parse_scene_json(raw)
    if parsed is None:
        print(f"\n{BAD} 解析不出 JSON。构建时这类会整张图跳过，主线一条都出不来。")
        print("  排查：改 prompts/vlm_select.txt 让模型只吐 JSON —— 改提示词不用动代码。")
        print("        小模型常见问题是加解释性前后缀，或把键名写错。")
        return 1
    if not parsed:
        # 【空选不是故障】。JSON 解析成功就说明提示词和输出格式没问题；
        # 挑不挑得中是模型看图后的判断。以前这里连同解析失败一起报 [失败]，
        # 会把人往「改提示词」的方向带，而实际上格式一点问题都没有。
        print(f"\n{OK} 格式通过：返回的是合法 JSON，键名也对")
        if n_real == 0:
            print("  模型一个都没挑 —— 用的是假标注，图里本来就没有这些目标，"
                  "这是正确行为。")
            print("  想连「挑目标」一起验，把 --image 指向 images_dir 里"
                  "有对应标注文件的图。")
        else:
            print(f"  但这张图的 {n_real} 个真实目标一个都没挑中。构建时这类图会跳过，"
                  "主线出不来。")
            print("  排查：多半是目标太小/太密模型看不清 —— 换一张目标大的图再试；"
                  "换几张都这样，说明这个模型的视觉能力撑不住这份数据。")
        return 0 if n_real == 0 else 1
    print(f"\n{OK} 解析成功，挑中 {len(parsed)} 个目标")
    for idx, info in list(parsed.items())[:2]:
        print(f"    [{idx}] attribute={info.get('attribute')!r} "
              f"color={info.get('color')!r}")
        print(f"         questions={info.get('questions')}")
        print(f"         description={info.get('description')!r}")
    missing = [k for k in ("attribute", "questions", "description")
               if not any((v.get(k) for v in parsed.values()))]
    if missing:
        print(f"  注意：{missing} 全为空。缺 questions 会回落模板问句；"
              "缺 description 时主线三个任务都出不来（主线要求描述齐全）。")

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

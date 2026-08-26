"""Qwen3.6 全模态接口 —— 上手前的三项验证。

不用手动转 base64、不用拼 curl，直接跑这个脚本即可。

用法：
    pip install requests
    python test_qwen.py 你的图片.jpg

脚本会依次验证：
  1. 纯文本能不能通（确认地址、key、模型名都对）
  2. 能不能收图片、会不会画框、坐标是什么格式
  3. 支不支持 logprobs（置信度）
"""

import base64
import json
import os
import sys

import requests

# ---------------- 改这里 ----------------
API_URL = "http://192.168.78.36:3012/v1/chat/completions"
API_KEY = "sk-bveYeVn6NAdRRElTWCqhtyJbkTL5XwweedczV9FJ05kDqhqX"
MODEL = "Qwen3.6-27B"
# 想检测什么就改这里
TARGET = "所有的人和车"
# ----------------------------------------

HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}


def post(payload, timeout=180):
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=timeout)
    return resp


def test_1_text():
    print("=" * 60)
    print("【测试1】纯文本连通性")
    print("=" * 60)
    payload = {
        "stream": False,
        "temperature": 0.6,
        "messages": [{"role": "user", "content": "请介绍一下自己"}],
        "chat_template_kwargs": {"enable_thinking": False},
        "model": MODEL,
    }
    try:
        r = post(payload, timeout=60)
    except Exception as e:
        print(f"✗ 请求失败：{e}")
        print("  → 检查网络能不能通到", API_URL)
        return False

    if r.status_code != 200:
        print(f"✗ HTTP {r.status_code}")
        print("  返回：", r.text[:500])
        print("  → 检查 API_KEY 和 MODEL 名称是否正确")
        return False

    content = r.json()["choices"][0]["message"]["content"]
    print("✓ 通了。模型回复前100字：")
    print(" ", content[:100].replace("\n", " "))
    return True


def test_2_image(image_path):
    print()
    print("=" * 60)
    print("【测试2】图片输入 + 画框能力（最关键的一步）")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"✗ 找不到图片：{image_path}")
        return False

    # 取图片真实宽高，告诉模型，让它按原图像素坐标输出
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            width, height = im.size
    except ImportError:
        print("  (未安装 pillow，无法自动读取图片尺寸，将使用占位值)")
        print("  建议 pip install pillow 后重跑，尺寸对结果影响很大")
        width, height = 640, 480
    except Exception as e:
        print(f"✗ 无法读取图片：{e}")
        return False

    print(f"  图片尺寸：{width} x {height}")

    # ---- 这就是"转base64"，就一行 ----
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    print(f"  base64 长度：{len(b64)} 字符（这就是为什么不适合手敲进 curl）")

    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else (ext or "jpeg")

    prompt = (
        f"这张图片的原始分辨率是宽{width}像素、高{height}像素。"
        f"请检测图中所有的{TARGET}，以JSON数组格式输出，"
        "每个元素包含两个字段：bbox_2d（格式为[x1,y1,x2,y2]，"
        f"必须是基于这张原图 {width}x{height} 像素坐标系的绝对像素坐标，"
        "左上角为原点(0,0)，x1<x2，y1<y2）和 label（物体类别名称）。"
        "只输出JSON本身，不要输出解释性文字，也不要用```包裹。"
    )

    payload = {
        "stream": False,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "model": MODEL,
    }

    try:
        r = post(payload)
    except Exception as e:
        print(f"✗ 请求失败：{e}")
        return False

    if r.status_code != 200:
        print(f"✗ HTTP {r.status_code}")
        print("  返回：", r.text[:800])
        print("  → 如果报错提到 image/content 格式，说明该部署可能没开多模态输入")
        return False

    content = r.json()["choices"][0]["message"]["content"]
    print("\n--- 模型原始输出 ---")
    print(content[:1500])
    print("--- 输出结束 ---\n")

    # 尝试解析并判断坐标格式
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        print("✗ 输出里没找到 JSON 数组 —— 模型没按格式返回，需要调 prompt")
        return False

    try:
        boxes = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        print(f"✗ JSON 解析失败：{e}")
        return False

    if not boxes:
        print("△ 解析成功，但没检测到目标。换一张有明显目标的图再试")
        return True

    print(f"✓ 成功解析出 {len(boxes)} 个框")

    # 判断坐标格式 —— 这是接下来最容易踩的坑
    all_vals = []
    for b in boxes:
        bb = b.get("bbox_2d") or b.get("bbox") or []
        all_vals.extend([abs(float(v)) for v in bb if isinstance(v, (int, float))])

    if not all_vals:
        print("△ 框里没有可识别的坐标字段，看看上面原始输出用的什么字段名")
        return True

    mx = max(all_vals)
    print(f"\n  坐标最大值：{mx}")
    if mx <= 1.001:
        print("  → 坐标格式：0~1 归一化相对坐标")
    elif mx > max(width, height) and mx <= 1000.5:
        print("  → 坐标格式：0~1000 千分比坐标")
    elif mx <= max(width, height) + 1:
        print("  → 坐标格式：绝对像素坐标（符合预期）")
    else:
        print("  → 坐标超出图片尺寸，格式异常，需要人工看一眼")

    print("\n  前3个框：")
    for b in boxes[:3]:
        print("   ", json.dumps(b, ensure_ascii=False))

    print("\n  ⚠ 重要：请务必把这些坐标画到图上肉眼确认对不对齐，")
    print("    数值格式看着对，不代表位置真的准。")
    return True


def test_3_logprobs():
    print()
    print("=" * 60)
    print("【测试3】logprobs 支持情况（置信度用）")
    print("=" * 60)
    payload = {
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "随便说一个两位数，只回答数字"}],
        "chat_template_kwargs": {"enable_thinking": False},
        "model": MODEL,
        "logprobs": True,
        "top_logprobs": 5,
    }
    try:
        r = post(payload, timeout=60)
    except Exception as e:
        print(f"✗ 请求失败：{e}")
        return

    if r.status_code != 200:
        print(f"✗ HTTP {r.status_code}：{r.text[:400]}")
        print("  → 如果提到 max_logprobs 超限，需要在 vLLM 启动参数加 --max-logprobs")
        print("  → 如果提到不认识该参数，可能 vLLM 版本较老")
        print("  → 不支持也没关系，服务里 ENABLE_LOGPROBS 保持 false 即可")
        return

    lp = r.json()["choices"][0].get("logprobs")
    if lp and lp.get("content"):
        print("✓ 支持 logprobs，可以把服务的 ENABLE_LOGPROBS 设成 true")
        first = lp["content"][0]
        print(f"  示例：token={first.get('token')!r} logprob={first.get('logprob')}")
    else:
        print("△ 请求成功但返回里没有 logprobs 字段，按不支持处理")
        print("  → 服务里 ENABLE_LOGPROBS 保持 false")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python test_qwen.py 你的图片.jpg")
        sys.exit(1)

    if not test_1_text():
        print("\n第一步就没通，先解决连通性问题，后面的测试没意义。")
        sys.exit(1)

    test_2_image(sys.argv[1])
    test_3_logprobs()

    print()
    print("=" * 60)
    print("三项测试跑完了。测试2 的结果最关键：")
    print("  - 能返回JSON坐标 → 可以继续搭服务")
    print("  - 返回一堆废话不给坐标 → 需要先调 prompt")
    print("  - 报错说不支持图片 → 该部署没开多模态，得先解决这个")
    print("=" * 60)

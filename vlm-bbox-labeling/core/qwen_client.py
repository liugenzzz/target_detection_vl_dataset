"""调用后端 Qwen 全模态模型的客户端。

统一封装：图片编码、请求组装、logprobs 开关、原始请求响应落盘。
"""

import base64
import json
import logging
import math
import os
import time
import uuid
from typing import Dict, Optional, Tuple

import requests

import config

logger = logging.getLogger(__name__)

# 后端如果不支持 logprobs，第一次失败后就不再重试，避免每张图都白跑一次
_logprobs_supported: Optional[bool] = None
# repetition_penalty 是 vLLM 扩展参数，老版本可能不认，同样做一次性探测
_rep_penalty_supported: Optional[bool] = None


def encode_image(image_bytes: bytes, mime: str = "jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def _save_raw_log(tag: str, payload: dict, response_text: str, elapsed: float):
    if not config.SAVE_RAW_LOG:
        return
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_{tag}_{uuid.uuid4().hex[:6]}.json"
        # 请求体里的 base64 图片非常大，落盘时替换掉，只留长度
        slim = json.loads(json.dumps(payload))
        for msg in slim.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        url = part["image_url"].get("url", "")
                        part["image_url"]["url"] = f"<base64 image, {len(url)} chars>"
        with open(os.path.join(config.LOG_DIR, fname), "w", encoding="utf-8") as f:
            json.dump(
                {"elapsed_sec": round(elapsed, 2), "request": slim, "response": response_text},
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:  # 日志失败不能影响主流程
        logger.warning("原始日志落盘失败：%s", e)


def call_model(
    prompt: str,
    image_data_url: Optional[str] = None,
    tag: str = "call",
    want_logprobs: Optional[bool] = None,
) -> Tuple[str, Dict]:
    """发一次请求。

    返回 (模型输出文本, meta信息字典)。
    meta 里含耗时、token用量、以及可用时的置信度参考值。
    """
    global _logprobs_supported, _rep_penalty_supported

    content = []
    if image_data_url:
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    content.append({"type": "text", "text": prompt})

    payload = {
        "stream": False,
        "temperature": config.QWEN_TEMPERATURE,
        "max_tokens": config.QWEN_MAX_TOKENS,
        "messages": [{"role": "user", "content": content}],
        "chat_template_kwargs": {"enable_thinking": config.QWEN_ENABLE_THINKING},
        "model": config.QWEN_MODEL,
    }

    if config.REPETITION_PENALTY and config.REPETITION_PENALTY != 1.0 and _rep_penalty_supported is not False:
        payload["repetition_penalty"] = config.REPETITION_PENALTY

    use_logprobs = config.ENABLE_LOGPROBS if want_logprobs is None else want_logprobs
    if use_logprobs and _logprobs_supported is not False:
        payload["logprobs"] = True
        payload["top_logprobs"] = config.TOP_LOGPROBS

    headers = {"Content-Type": "application/json"}
    if config.QWEN_API_KEY:
        headers["Authorization"] = f"Bearer {config.QWEN_API_KEY}"

    start = time.time()
    resp = requests.post(config.QWEN_API_URL, headers=headers, json=payload, timeout=config.QWEN_TIMEOUT)

    # 后端不认某个可选参数时自动降级重试，不让整个请求失败
    if resp.status_code >= 400 and ("logprobs" in payload or "repetition_penalty" in payload):
        logger.warning("带可选参数的请求失败(%s)，自动降级重试。响应：%s", resp.status_code, resp.text[:300])
        if "repetition_penalty" in payload:
            _rep_penalty_supported = False
            payload.pop("repetition_penalty", None)
        if "logprobs" in payload:
            _logprobs_supported = False
            payload.pop("logprobs", None)
            payload.pop("top_logprobs", None)
        resp = requests.post(config.QWEN_API_URL, headers=headers, json=payload, timeout=config.QWEN_TIMEOUT)

    resp.raise_for_status()
    elapsed = time.time() - start
    data = resp.json()

    choice = data["choices"][0]
    text = choice["message"]["content"] or ""

    meta = {
        "elapsed_sec": round(elapsed, 2),
        "usage": data.get("usage"),
        "finish_reason": choice.get("finish_reason"),
    }

    # 输出被截断是实验中很常见的坑，单独标出来
    if choice.get("finish_reason") == "length":
        meta["truncated"] = True
        logger.warning("模型输出被 max_tokens 截断，考虑调大 QWEN_MAX_TOKENS")

    lp = choice.get("logprobs")
    if lp:
        _logprobs_supported = True
        meta["logprobs_available"] = True
        meta["token_logprobs"] = _extract_token_probs(lp)
    else:
        meta["logprobs_available"] = False

    _save_raw_log(tag, payload, text, elapsed)
    return text, meta


def _extract_token_probs(logprobs_obj):
    """把 logprobs 拍平成 [(token, 概率), ...]，供后续算置信度用。"""
    out = []
    items = logprobs_obj.get("content") or []
    for item in items:
        token = item.get("token")
        lp = item.get("logprob")
        if token is None or lp is None:
            continue
        try:
            out.append({"token": token, "prob": round(math.exp(lp), 4)})
        except OverflowError:
            continue
    return out


def confidence_for_number(token_probs, number_str: str, start_at: int = 0):
    """在 token 概率序列里找某个数字（比如 class_id）对应的 token，取其概率作为置信度参考。

    start_at 是搜索起点：同一张图里多个框可能是同一个 class_id，
    必须按顺序往后找，否则每个框都会匹配到第一次出现的那个 token，
    导致所有框的置信度完全相同（这是个实际踩过的坑）。

    返回 (置信度, 下一次搜索的起点)。找不到时返回 (None, start_at)。

    注意：这是个近似值。数字可能被切成多个 token，这里取组成它的 token 概率的乘积。
    这个值比让模型自己"说"一个置信度可靠，但仍然只能当参考，不能当统计意义上的准确率。
    """
    if not token_probs or not number_str:
        return None, start_at

    target = str(number_str).strip()
    n = len(token_probs)

    i = max(0, start_at)
    while i < n:
        buf = ""
        prob = 1.0
        j = i
        while j < n:
            tok = token_probs[j]["token"].strip()
            if not tok:
                j += 1
                continue
            if not target.startswith(buf + tok):
                break
            buf += tok
            prob *= token_probs[j]["prob"]
            j += 1
            if buf == target:
                # 匹配成功，下次从这个位置之后继续找
                return round(prob, 4), j
        i += 1

    return None, start_at

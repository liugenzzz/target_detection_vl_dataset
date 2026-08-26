"""调自建 Qwen 服务，为目标生成【视觉指代】和【描述语句】。

一次调用同时拿两样东西 —— 所以给同类密集目标补视觉指代不额外增加成本。

三个部署要点：
  1. 结果按图落盘到 cache_dir，支持断点续跑。两万张图跑一遍要几小时，
     中途挂掉从头再来是不可接受的。
  2. vlm.enabled=false 时全部回落到模板，不发任何请求 —— 本地没有服务时
     也能把管道跑通。
  3. 服务不可达或返回不可解析时，单条失败回落模板，不中断整批。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class VlmResult:
    referring: str
    description: str
    source: str          # "vlm" | "template" | "cache"


class VlmClient:
    def __init__(self, cfg):
        v = cfg.get_path("vlm", {}) or {}
        self.enabled = bool(v.get("enabled", True))
        self.api_url = str(v.get("api_url", ""))
        self.api_key = str(v.get("api_key", ""))
        self.model = str(v.get("model", ""))
        self.timeout = int(v.get("timeout", 300))
        self.temperature = float(v.get("temperature", 0.35))
        self.max_tokens = int(v.get("max_tokens", 1024))
        self.max_retries = int(v.get("max_retries", 3))
        cache = v.get("cache_dir") or ""
        self.cache_dir = Path(cache) if cache else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"vlm": 0, "cache": 0, "template": 0, "failed": 0}

    # ------------------------------------------------------------ 缓存
    def _cache_path(self, image_path: Path, bbox: Sequence[int]) -> Optional[Path]:
        if not self.cache_dir:
            return None
        key = hashlib.md5(f"{image_path.name}|{list(bbox)}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, path: Optional[Path]) -> Optional[VlmResult]:
        if not path or not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return VlmResult(d.get("referring", ""), d.get("description", ""), "cache")
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, path: Optional[Path], result: VlmResult) -> None:
        if not path:
            return
        try:
            path.write_text(json.dumps(
                {"referring": result.referring, "description": result.description},
                ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            logger.warning("写缓存失败 %s：%s", path, exc)

    # ------------------------------------------------------------ 调用
    def describe(self, image_path: Path, bbox: Sequence[int], label: str,
                 prompt_text: str, fallback: VlmResult) -> VlmResult:
        """给一个目标生成指代与描述。任何失败都回落 fallback，不抛异常。"""
        cache_path = self._cache_path(image_path, bbox)
        cached = self._read_cache(cache_path)
        if cached:
            self.stats["cache"] += 1
            return self._fill_blanks(cached, fallback)

        if not self.enabled:
            self.stats["template"] += 1
            return fallback

        result = self._request(image_path, prompt_text)
        if result is None:
            self.stats["failed"] += 1
            return fallback

        result = self._fill_blanks(result, fallback)
        self.stats["vlm"] += 1
        self._write_cache(cache_path, result)
        return result

    @staticmethod
    def _fill_blanks(result: VlmResult, fallback: VlmResult) -> VlmResult:
        """VLM 某一项返回空时，用模板补上那一项。"""
        return VlmResult(
            referring=result.referring or fallback.referring,
            description=result.description or fallback.description,
            source=result.source,
        )

    def _request(self, image_path: Path, prompt_text: str) -> Optional[VlmResult]:
        try:
            import requests
        except ImportError:
            logger.error("未安装 requests，无法调用 VLM 服务")
            return None

        try:
            b64 = base64.b64encode(image_path.read_bytes()).decode()
        except OSError as exc:
            logger.warning("读图失败 %s：%s", image_path, exc)
            return None

        mime = "png" if image_path.suffix.lower() == ".png" else "jpeg"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                {"type": "text", "text": prompt_text},
            ]}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(self.api_url, json=payload,
                                     headers=headers, timeout=self.timeout)
                if resp.status_code != 200:
                    logger.warning("VLM 返回 HTTP %s（第 %s 次）", resp.status_code, attempt)
                    time.sleep(min(2 ** attempt, 10))
                    continue
                text = resp.json()["choices"][0]["message"]["content"]
                return _parse_vlm_json(text)
            except Exception as exc:                     # noqa: BLE001
                logger.warning("调用 VLM 失败（第 %s 次）：%s", attempt, exc)
                time.sleep(min(2 ** attempt, 10))
        return None


def _parse_vlm_json(text: str) -> Optional[VlmResult]:
    """从模型输出里抠出 JSON。模型常会用 ```json 包裹或加解释性前后缀。"""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return VlmResult(
        referring=str(data.get("referring") or "").strip(),
        description=str(data.get("description") or "").strip(),
        source="vlm",
    )

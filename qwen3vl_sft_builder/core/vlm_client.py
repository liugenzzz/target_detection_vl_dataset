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
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class FatalVlmError(RuntimeError):
    """配置层面的错误（认证失败、模型名不对、路径不对）。

    这类错误重试多少次都不会成功，而且必然对【所有】请求成立 ——
    十万条任务全都会失败。所以一旦出现就立即中止整批，别让几十万次
    注定失败的请求砸向服务。
    """


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
        self.concurrency = max(1, int(v.get("concurrency", 4)))
        # 挑对象那次调用要顺带生成多样的问句，温度低了三句会写得几乎一样。
        self.temperature_select = float(v.get("temperature_select", 0.85))
        cache = v.get("cache_dir") or ""
        self.cache_dir = Path(cache) if cache else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"vlm": 0, "cache": 0, "template": 0, "failed": 0, "prefetched": 0}
        self._lock = threading.Lock()      # 保护 stats；缓存写入靠原子 rename
        # prefetch 跑完后置位。此后 describe() 遇到缓存未命中不再重新发请求 ——
        # 组装阶段是串行的，那里发请求会带着重试和超时把整批任务拖垮
        # （10 万条里 1% 预取失败 = 1000 次串行调用）。预取没拿到的直接回落模板。
        self._prefetch_done = False
        # 预取结果的内存副本。磁盘缓存是「跨进程复用」，内存是「本次运行的取用通道」——
        # 只靠磁盘的话，cache_dir 未配置或写盘失败时预取结果会被静默丢弃，
        # 白烧一整轮 API 却产出纯模板数据集。
        self._memory: Dict[str, VlmResult] = {}
        self._cancel = threading.Event()
        self._fatal: Optional[str] = None      # 命中配置错误后记在这里，供 prefetch 中止

    # ------------------------------------------------------------ 缓存
    @staticmethod
    def _key(image_path: Path, bbox: Sequence[int]) -> str:
        return hashlib.md5(f"{image_path.name}|{list(bbox)}".encode()).hexdigest()

    def _cache_path(self, image_path: Path, bbox: Sequence[int]) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{self._key(image_path, bbox)}.json"

    def _read_cache(self, path: Optional[Path]) -> Optional[VlmResult]:
        if not path or not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return VlmResult(d.get("referring", ""), d.get("description", ""), "cache")
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, path: Optional[Path], result: VlmResult) -> None:
        """原子写入：先写临时文件再 rename。并发下多个线程写不同 key，
        rename 在同一文件系统内是原子的，中途被 Ctrl-C 也不会留下半截 JSON。"""
        if not path:
            return
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps(
                {"referring": result.referring, "description": result.description},
                ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("写缓存失败 %s：%s", path, exc)
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------ 并发预取
    def prefetch(self, tasks: Sequence[Tuple[Path, Sequence[int], str, str]],
                 progress_every: int = 200) -> None:
        """并发把所有目标的 VLM 结果灌进缓存，之后 describe() 全部命中缓存。

        tasks 每项为 (image_path, bbox, label, prompt_text)。

        并发用线程而非进程：VLM 调用是纯 I/O 等待（网络往返数秒），
        requests 阻塞期间会释放 GIL，线程能真正并行；多进程还要付出
        图片 base64 跨进程传输的代价。

        已缓存的直接跳过 —— 中途 Ctrl-C 后重跑只补没跑完的部分。
        """
        if not self.enabled or not tasks:
            return
        self._prefetch_done = True

        todo = [t for t in tasks
                if self._key(t[0], t[1]) not in self._memory
                and not self._read_cache(self._cache_path(t[0], t[1]))]
        hit = len(tasks) - len(todo)
        if hit:
            logger.info("VLM 缓存命中 %d 条，需要新调用 %d 条", hit, len(todo))
        if not todo:
            logger.info("全部命中缓存，跳过 VLM 调用")
            return

        logger.info("开始并发调用 VLM：%d 条，并发 %d 路", len(todo), self.concurrency)
        started = time.time()
        done = 0

        def work(task):
            image_path, bbox, _label, prompt_text = task
            # 已请求中断，或已命中配置错误 —— 尚未启动的任务直接跳过
            if self._cancel.is_set() or self._fatal:
                return None
            raw = self._request_raw(image_path, prompt_text, self.temperature_select)
            if raw is None:
                return None
            key = self._key(image_path, bbox)
            with self._lock:
                self._memory[key] = raw        # 内存优先，磁盘只是跨进程复用
            self._write_cache(self._cache_path(image_path, bbox),
                              VlmResult("", raw, "vlm"))
            return key

        # 不用 with：ThreadPoolExecutor 的 __exit__ 是 shutdown(wait=True)，
        # 会先把整个队列跑完才让 KeyboardInterrupt 冒出来 —— 10 万条任务时
        # 第一次 Ctrl-C 要等几小时才生效，和「支持断点续跑」自相矛盾。
        pool = ThreadPoolExecutor(max_workers=self.concurrency)
        try:
            futures = [pool.submit(work, t) for t in todo]
            for fut in as_completed(futures):
                done += 1
                try:
                    ok = fut.result() is not None
                except Exception as exc:                # noqa: BLE001
                    logger.warning("预取任务异常：%s", exc)
                    ok = False
                with self._lock:
                    self.stats["prefetched" if ok else "failed"] += 1
                if self._fatal:
                    self._cancel.set()
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise FatalVlmError(self._fatal)
                if done % progress_every == 0 or done == len(todo):
                    elapsed = time.time() - started
                    rate = done / elapsed if elapsed else 0
                    left = (len(todo) - done) / rate if rate else 0
                    logger.info("VLM 预取 %d/%d（%.1f 条/秒，剩余约 %.1f 分钟）",
                                done, len(todo), rate, left / 60)
        except KeyboardInterrupt:
            self._cancel.set()
            pool.shutdown(wait=False, cancel_futures=True)
            logger.warning("收到中断：已完成 %d/%d，结果已落盘，重跑从断点继续",
                           done, len(todo))
            raise
        else:
            pool.shutdown(wait=True)
        finally:
            failed = self.stats["failed"]
            if failed:
                logger.warning("预取有 %d 条失败，这些目标的描述会回落模板。"
                               "失败率高说明服务不稳或 prompts/vlm_describe.txt "
                               "需要调整，建议先排查再全量跑。", failed)

    # ------------------------------------------------------------ 调用
    def scene_info(self, image_path: Path, key: Sequence[int],
                   valid_indices) -> Dict[int, Dict[str, str]]:
        """取该图「挑对象」调用的结果。未启用 / 未命中 / 解析失败时返回空 dict，
        依赖它的任务（ground_attribute、attribute_qa、image_caption）
        会因条件不满足而跳过，不影响其余任务。

        只保留仍在 valid_indices 里的编号 —— 模型可能返回被质量过滤掉的框，
        或者干脆编一个不存在的编号。
        """
        raw = self._memory.get(self._key(image_path, key))
        if raw is None:
            cached = self._read_cache(self._cache_path(image_path, key))
            raw = cached.description if cached else None
        if not raw:
            return {}
        parsed = _parse_scene_json(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            return {}
        return {i: v for i, v in parsed.items() if i in valid_indices}

    def describe(self, image_path: Path, bbox: Sequence[int], label: str,
                 prompt_text: str, fallback: VlmResult) -> VlmResult:
        """给一个目标生成指代与描述。任何失败都回落 fallback，不抛异常。"""
        key = self._key(image_path, bbox)
        in_mem = self._memory.get(key)
        if in_mem:
            with self._lock:
                self.stats["cache"] += 1
            return self._fill_blanks(in_mem, fallback)

        cache_path = self._cache_path(image_path, bbox)
        cached = self._read_cache(cache_path)
        if cached:
            with self._lock:
                self.stats["cache"] += 1
            return self._fill_blanks(cached, fallback)

        if not self.enabled or self._prefetch_done:
            with self._lock:
                self.stats["template"] += 1
            return fallback

        result = self._request(image_path, prompt_text)
        if result is None:
            with self._lock:
                self.stats["failed"] += 1
            return fallback

        result = self._fill_blanks(result, fallback)
        with self._lock:
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

    def _request_raw(self, image_path: Path, prompt_text: str) -> Optional[str]:
        """发一次请求，返回模型输出的原始文本（不解析）。"""
        payload = self._payload(image_path, prompt_text)
        if payload is None:
            return None
        return self._post(payload)

    def _payload(self, image_path: Path, prompt_text: str,
                 temperature: Optional[float] = None) -> Optional[dict]:
        """构造 OpenAI 兼容的多模态请求体。图片以 base64 data URI 传入。"""
        try:
            b64 = base64.b64encode(image_path.read_bytes()).decode()
        except OSError as exc:
            logger.warning("读图失败 %s：%s", image_path, exc)
            return None
        mime = "png" if image_path.suffix.lower() == ".png" else "jpeg"
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                {"type": "text", "text": prompt_text},
            ]}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }

    def _post(self, payload: dict) -> Optional[str]:
        """发请求并返回模型输出的原始文本。配置类错误记入 _fatal 并立即放弃重试。"""
        try:
            import requests
        except ImportError:
            logger.error("未安装 requests，无法调用 VLM 服务")
            return None

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(self.api_url, json=payload,
                                     headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                # 4xx 里除了 429（限流）都是配置问题，重试没有意义 ——
                # 请求本身就不对，再发一百次还是同样的错。
                hint = _diagnose(resp.status_code, resp.text)
                if hint:
                    with self._lock:
                        if self._fatal is None:
                            self._fatal = hint
                    return None
                logger.warning("VLM 返回 HTTP %s（第 %s 次）", resp.status_code, attempt)
            except Exception as exc:                     # noqa: BLE001
                logger.warning("调用 VLM 失败（第 %s 次）：%s", attempt, exc)
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 10))
        return None

    def _request_raw(self, image_path: Path, prompt_text: str,
                     temperature: Optional[float] = None) -> Optional[str]:
        """发一次请求，返回模型输出的原始文本（不解析）。"""
        payload = self._payload(image_path, prompt_text, temperature)
        return self._post(payload) if payload else None

    def _request(self, image_path: Path, prompt_text: str) -> Optional[VlmResult]:
        raw = self._request_raw(image_path, prompt_text)
        return _parse_vlm_json(raw) if raw else None


def _diagnose(status: int, body: str) -> Optional[str]:
    """判断这个 HTTP 状态码是不是「重试也没用」的配置错误。

    是则返回一句能直接照做的提示，否则返回 None（表示可以重试）。
    429 限流和 5xx 属于临时故障，要重试。
    """
    if status in (429,) or status >= 500:
        return None
    snippet = (body or "")[:200]
    if status == 401:
        return ("HTTP 401 认证失败：api_key 没设置或不对。\n"
                "    PowerShell:  $env:VLM_API_KEY=\"sk-...\"\n"
                "    Linux/Mac :  export VLM_API_KEY=sk-...\n"
                "    或写进 config/local.yaml 的 vlm.api_key")
    if status == 403:
        return "HTTP 403 无权访问：api_key 对但没有这个模型的权限，找服务方确认。"
    if status == 404:
        return ("HTTP 404 路径不对：vlm.api_url 必须带完整路径，"
                "形如 http://主机:端口/v1/chat/completions")
    if status in (400, 422):
        return (f"HTTP {status} 请求被拒：多半是 vlm.model 的模型名和服务上的对不上，"
                f"或该部署不支持图片输入。\n    服务返回：{snippet}")
    return f"HTTP {status}：{snippet}"


def _parse_scene_json(text: str) -> Optional[Dict[int, Dict[str, str]]]:
    """解析「挑对象」调用的返回：{"picked":[{"id":0,"attribute":..,"color":..,"description":..}]}

    返回 {box_index: {attribute, color, description}}。解析不出返回 None。
    """
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    picked = data.get("picked") if isinstance(data, dict) else None
    if not isinstance(picked, list):
        return None
    out: Dict[int, Dict[str, str]] = {}
    for item in picked:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        qs = item.get("questions")
        questions = [str(q).strip() for q in qs if str(q).strip()] if isinstance(qs, list) else []
        out[idx] = {
            "attribute": str(item.get("attribute") or "").strip(),
            "color": str(item.get("color") or "").strip(),
            "description": str(item.get("description") or "").strip(),
            "questions": questions,
        }
    return out or None


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
    referring = str(data.get("referring") or "").strip()
    description = str(data.get("description") or "").strip()
    if not referring and not description:
        # 形状不对的 JSON（键名写错等）会解析成两个空串。若当成功处理，
        # 会写入一条空缓存，之后每次运行都命中它，把该目标永久钉死在模板文本上，
        # 而报告里还显示成功。
        return None
    return VlmResult(referring=referring, description=description, source="vlm")

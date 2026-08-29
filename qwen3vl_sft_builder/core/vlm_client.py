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

from . import progress

logger = logging.getLogger(__name__)


class FatalVlmError(RuntimeError):
    """配置层面的错误（认证失败、模型名不对、路径不对）。

    这类错误重试多少次都不会成功，而且必然对【所有】请求成立 ——
    十万条任务全都会失败。所以一旦出现就立即中止整批，别让几十万次
    注定失败的请求砸向服务。
    """


@dataclass
class Endpoint:
    """模型池里的一路。多路的意义有两个：吞吐（十万条样本要跑几小时）
    和容错（一路挂了整批不必停）。"""
    name: str
    api_url: str
    api_key: str
    model: str
    concurrency: int
    fatal: Optional[str] = None     # 该路的配置错误（401、模型名不对……）

    @property
    def healthy(self) -> bool:
        return self.fatal is None


def _endpoints_from(v: dict) -> List[Endpoint]:
    """解析模型池配置。写了 vlm.endpoints 就用池子，没写就把平铺的
    api_url/model 当成只有一路的池子 —— 老配置不用改。"""
    raw = v.get("endpoints") or []
    if not raw:
        raw = [{"api_url": v.get("api_url", ""), "api_key": v.get("api_key", ""),
                "model": v.get("model", ""), "concurrency": v.get("concurrency", 4)}]
    out = []
    for i, e in enumerate(raw):
        if not isinstance(e, dict):
            raise ValueError(f"vlm.endpoints[{i}] 必须是字典，实为 {type(e).__name__}")
        out.append(Endpoint(
            name=str(e.get("name") or f"{e.get('model') or '?'}@{e.get('api_url', '')[:40]}"),
            api_url=str(e.get("api_url") or v.get("api_url", "")),
            api_key=str(e.get("api_key") if e.get("api_key") is not None
                        else v.get("api_key", "")),
            model=str(e.get("model") or v.get("model", "")),
            concurrency=max(1, int(e.get("concurrency", v.get("concurrency", 4)))),
        ))
    return out


@dataclass
class VlmResult:
    """一次调用的返回。description 存的是模型输出的【原始文本】，
    怎么解析由调用方决定 —— 挑对象那次要解析成 {框号: 属性}，
    质检那次要解析成评分，客户端本身不关心。"""
    description: str
    source: str          # "vlm" | "cache"


def _prompt_fingerprint() -> str:
    """prompts/ 下所有 .txt 的内容指纹。

    按内容而不是按修改时间 —— git checkout 会改 mtime 但内容没变，
    按时间算会把好好的缓存全废掉。
    """
    import prompts as _p
    h = hashlib.md5()
    for path in sorted(_p.PROMPT_DIR.rglob("*.txt")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


class VlmClient:
    def __init__(self, cfg, role: str = ""):
        """role 非空时，先看 vlm.roles.<role> 有没有单独的配置，有就用它覆盖。

        审核这一步尤其需要：用生成它的同一个模型来审自己写的描述，等于自己
        给自己打分，它会倾向于认同自己的输出。有第二个模型时应该让审核走那个。
        """
        v = dict(cfg.get_path("vlm", {}) or {})
        if role:
            override = (v.get("roles") or {}).get(role) or {}
            if override:
                v = {**v, **override}
                # roles.<role> 里写了 endpoints 就换池子；只写了 model 则沿用
                # 外层地址换个模型名，这时要把外层的 endpoints 清掉免得盖不住。
                if override.get("endpoints") is None and override.get("model"):
                    v["endpoints"] = []
        self.role = role
        self.enabled = bool(v.get("enabled", True))
        self.api_url = str(v.get("api_url", ""))
        self.api_key = str(v.get("api_key", ""))
        self.model = str(v.get("model", ""))
        self.timeout = int(v.get("timeout", 300))
        self.temperature = float(v.get("temperature", 0.35))
        self.max_tokens = int(v.get("max_tokens", 1024))
        self.max_retries = int(v.get("max_retries", 3))
        # 模型池。总并发是各路之和 —— 每一路自己的 concurrency 是它扛得住的量，
        # 加起来才是这台机器能同时压出去的请求数。
        self.endpoints = _endpoints_from(v)
        self.concurrency = sum(e.concurrency for e in self.endpoints)
        self._rr = 0                      # 轮转游标，_lock 保护
        # prompts/ 全部内容的指纹，进缓存键。改提示词自动让缓存失效。
        self._prompt_fp = _prompt_fingerprint()
        # 流式：一次调用要几十秒，非流式时它和「卡死」长得一模一样。
        self.stream = bool(v.get("stream", True))
        self._on_token = None          # 进度条挂进来，收到 token 就刷新
        self._last_stream_status = None
        self.show_progress = bool(v.get("progress", True))
        # 加权轮转表：concurrency=8 的那路在表里出现 8 次
        self._rotation = [e for e in self.endpoints for _ in range(e.concurrency)]
        # 挑对象那次调用要顺带生成多样的问句，温度低了三句会写得几乎一样。
        self.temperature_select = float(v.get("temperature_select", 0.85))
        # 改写要保真不要发挥，用低温
        cache = v.get("cache_dir") or ""
        # 按角色分子目录。两个理由：
        #   1. 缓存键是 (图, 一串整数)，构建用的是 bbox / 图片尺寸，质检用的是
        #      问答对的哈希 —— 同一个键空间里理论上会撞。
        #   2. 换了质检模型只想重跑质检，删一个子目录就行，不必连构建结果一起废。
        self.cache_dir = Path(cache) / role if (cache and role) else (
            Path(cache) if cache else None)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"vlm": 0, "cache": 0, "template": 0, "failed": 0, "prefetched": 0}
        self.by_endpoint = {e.name: 0 for e in self.endpoints}
        self._lock = threading.Lock()      # 保护 stats；缓存写入靠原子 rename
        # prefetch 跑完后置位。此后 describe() 遇到缓存未命中不再重新发请求 ——
        # 组装阶段是串行的，那里发请求会带着重试和超时把整批任务拖垮
        # （10 万条里 1% 预取失败 = 1000 次串行调用）。预取没拿到的直接回落模板。
        self._prefetch_done = False
        # 预取结果的内存副本。磁盘缓存是「跨进程复用」，内存是「本次运行的取用通道」——
        # 只靠磁盘的话，cache_dir 未配置或写盘失败时预取结果会被静默丢弃，
        # 白烧一整轮 API 却产出纯模板数据集。
        # 存的是模型输出的【原始文本】，跟磁盘缓存里 description 字段一致。
        # 解析放到取用时做 —— 挑对象和质检解析成不同的形状。
        self._memory: Dict[str, str] = {}
        self._cancel = threading.Event()
        self._fatal: Optional[str] = None      # 命中配置错误后记在这里，供 prefetch 中止

    # ------------------------------------------------------------ 缓存
    def _key(self, image_path: Path, bbox: Sequence[int]) -> str:
        """缓存键 = 图片 + 参数 + 【提示词指纹】。

        提示词必须进键。缓存的意义是「同一个问题不重复问」，改了提示词就
        不是同一个问题了 —— 少了这一项，你改完 prompts/ 重跑会全部命中旧缓存，
        看到的还是上一版的结果，而且完全没有提示。
        实测踩过：改完描述子类型重跑三次，输出一字未变。
        """
        return hashlib.md5(
            f"{image_path.name}|{list(bbox)}|{self._prompt_fp}".encode()).hexdigest()

    def _cache_path(self, image_path: Path, bbox: Sequence[int]) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{self._key(image_path, bbox)}.json"

    def _read_cache(self, path: Optional[Path]) -> Optional[VlmResult]:
        if not path or not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return VlmResult(d.get("description", ""), "cache")
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
                {"description": result.description},
                ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("写缓存失败 %s：%s", path, exc)
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------ 并发预取
    def prefetch(self, tasks: Sequence[Tuple[Path, Sequence[int], str, str]],
                 label: str = "VLM 预取",
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
                              VlmResult(raw, "vlm"))
            return key

        # 不用 with：ThreadPoolExecutor 的 __exit__ 是 shutdown(wait=True)，
        # 会先把整个队列跑完才让 KeyboardInterrupt 冒出来 —— 10 万条任务时
        # 第一次 Ctrl-C 要等几小时才生效，和「支持断点续跑」自相矛盾。
        pool = ThreadPoolExecutor(max_workers=self.concurrency)
        bar = progress.make(label, len(todo), self.show_progress)
        # 流式的 token 回调挂到进度条上：长调用途中也在动，卡住一眼看得出来
        tokens = [0]

        def on_token(n):
            tokens[0] += n

        self._on_token = on_token
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
                failed = self.stats["failed"]
                bar.step(note=(f"失败 {failed}" if failed else "") or
                              (f"{tokens[0] // 1000}k tokens" if tokens[0] else ""))
                if self._fatal:
                    self._cancel.set()
                    pool.shutdown(wait=False, cancel_futures=True)
                    bar.close()
                    raise FatalVlmError(self._fatal)
        except KeyboardInterrupt:
            self._cancel.set()
            pool.shutdown(wait=False, cancel_futures=True)
            bar.close()
            logger.warning("收到中断：已完成 %d/%d，结果已落盘，重跑从断点继续",
                           done, len(todo))
            raise
        else:
            pool.shutdown(wait=True)
            bar.close()
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
        raw = self.raw_result(image_path, key)
        if not raw:
            return {}
        parsed = _parse_scene_json(raw)
        if not parsed:
            return {}
        return {i: v for i, v in parsed.items() if i in valid_indices}



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

    def _post_stream(self, requests, ep, body, headers) -> Optional[str]:
        """流式发一次请求，边收边攒。返回完整文本；非 200 返回 None 走原路重试。

        流式的意义不是快，是【看得见】：一次调用要几十秒，非流式时它和「卡死」
        长得一模一样。流式下每收到一个 token 就刷新一次「还活着」，
        真卡住时也能从「多久没收到新 token」判出来，而不是干等超时。
        """
        payload = dict(body, stream=True)
        try:
            with requests.post(ep.api_url, json=payload, headers=headers,
                               timeout=self.timeout, stream=True) as resp:
                if resp.status_code != 200:
                    # 让调用方按非流式那套逻辑处理状态码
                    self._last_stream_status = (resp.status_code, resp.text[:400])
                    return None
                # 【必须自己按 UTF-8 解码】。iter_lines(decode_unicode=True) 用的是
                # resp.encoding，而服务端 Content-Type 不带 charset 时 requests
                # 默认 ISO-8859-1 —— 中文会变成「ç\x99½è\x89²」这种乱码，
                # 而且不报错、一路写进数据集。实测踩到。
                chunks, last = [], time.time()
                for raw_bytes in resp.iter_lines(decode_unicode=False):
                    if not raw_bytes:
                        continue
                    raw = raw_bytes.decode("utf-8", errors="replace")
                    line = raw[6:] if raw.startswith("data: ") else raw
                    if line.strip() in ("[DONE]", ""):
                        continue
                    try:
                        piece = json.loads(line)["choices"][0]
                    except (ValueError, KeyError, IndexError):
                        continue
                    delta = (piece.get("delta") or {}).get("content") or ""
                    if delta:
                        chunks.append(delta)
                        now = time.time()
                        if now - last > 1.0:      # 每秒最多回调一次，别刷爆
                            last = now
                            if self._on_token:
                                self._on_token(len(delta))
                return "".join(chunks) or None
        except Exception as exc:                       # noqa: BLE001
            logger.warning("%s 流式读取中断：%s", ep.name, exc)
            return None

    def _pick(self) -> Optional[Endpoint]:
        """轮转取一路健康的端点。全都挂了返回 None。

        按各路的 concurrency 加权 —— 一路写 8、一路写 3，说明前者扛得住的量
        是后者的两倍多。均分会把慢的那路压垮，快的那路闲着。
        """
        with self._lock:
            healthy = [e for e in self._rotation if e.healthy]
            if not healthy:
                return None
            ep = healthy[self._rr % len(healthy)]
            self._rr += 1
            return ep

    def _post(self, payload: dict) -> Optional[str]:
        """发请求并返回模型输出的原始文本。

        每次重试换一路端点：多路的时候，一路 401、一路超时，轮着试还能跑完；
        单路的时候行为和以前一样。配置类错误只标记【那一路】，全部端点都挂了
        才记入 _fatal 中止整批 —— 否则一路配错就把好的几路也一起停了。
        """
        try:
            import requests
        except ImportError:
            logger.error("未安装 requests，无法调用 VLM 服务")
            return None

        for attempt in range(1, self.max_retries + 1):
            ep = self._pick()
            if ep is None:
                with self._lock:
                    if self._fatal is None:
                        self._fatal = ("模型池里所有端点都不可用：\n    "
                                       + "\n    ".join(f"{e.name} -> {e.fatal}"
                                                       for e in self.endpoints))
                return None

            headers = {"Content-Type": "application/json"}
            if ep.api_key:
                headers["Authorization"] = f"Bearer {ep.api_key}"
            body = dict(payload, model=ep.model)     # 模型名由这一路决定

            try:
                if self.stream:
                    text = self._post_stream(requests, ep, body, headers)
                    if text is not None:
                        with self._lock:
                            self.by_endpoint[ep.name] = self.by_endpoint.get(ep.name, 0) + 1
                        return text
                    resp = None
                else:
                    resp = requests.post(ep.api_url, json=body,
                                         headers=headers, timeout=self.timeout)
                if resp is None:
                    logger.warning("%s 流式返回为空（第 %s 次）", ep.name, attempt)
                    if attempt < self.max_retries:
                        time.sleep(min(2 ** attempt, 10))
                    continue
                if resp.status_code == 200:
                    with self._lock:
                        self.by_endpoint[ep.name] = self.by_endpoint.get(ep.name, 0) + 1
                    return resp.json()["choices"][0]["message"]["content"]
                # 4xx 里除了 429（限流）都是配置问题，重试没有意义 ——
                # 请求本身就不对，再发一百次还是同样的错。
                hint = _diagnose(resp.status_code, resp.text)
                if hint:
                    with self._lock:
                        if ep.fatal is None:
                            ep.fatal = hint
                            logger.error("模型池中 %s 已摘除：%s", ep.name, hint)
                    continue                          # 换下一路再试
                logger.warning("%s 返回 HTTP %s（第 %s 次）",
                               ep.name, resp.status_code, attempt)
            except Exception as exc:                     # noqa: BLE001
                logger.warning("调用 %s 失败（第 %s 次）：%s", ep.name, attempt, exc)
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 10))
        return None


    def text_result(self, key_src: str) -> Optional[str]:
        """取一条纯文本调用的结果。"""
        k = self._key(Path(key_src), [])
        raw = self._memory.get(k)
        if raw is None:
            cached = self._read_cache(self._cache_path(Path(key_src), []))
            raw = cached.description if cached else None
        return raw

    def raw_result(self, image_path: Path, key) -> Optional[str]:
        """取一条图片调用的原始返回。"""
        k = self._key(image_path, key)
        raw = self._memory.get(k)
        if raw is None:
            cached = self._read_cache(self._cache_path(image_path, key))
            raw = cached.description if cached else None
        return raw

    def _request_raw(self, image_path: Path, prompt_text: str,
                     temperature: Optional[float] = None) -> Optional[str]:
        """发一次请求，返回模型输出的原始文本（不解析）。"""
        payload = self._payload(image_path, prompt_text, temperature)
        return self._post(payload) if payload else None



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
        # 403 和 401 是两回事：能返回这种结构化错误说明 key 有效、服务认得它，
        # 只是这个 token 的授权范围里没有 vlm.model 指的那个模型。
        # 网关（new-api / one-api 这类）按模型分组发 token，很常见。
        return ("HTTP 403 无权访问：api_key 是【有效】的，但它没有 vlm.model "
                "指定的那个模型的权限。\n"
                "    要么模型名写错了，要么这个 key 的授权里没有这个模型。\n"
                f"    服务返回：{snippet}\n"
                "    先看这个 key 到底能用哪些模型：\n"
                "        python scripts/check_vlm.py --list-models")
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
            # 第三轮的问句由模型和 description 成对生成 —— 问句问了哪几样、
            # 答句就答哪几样。模型没给就回落 ask_describe 问法池。
            "describe_q": str(item.get("describe_q") or "").strip(),
            "questions": questions,
        }
    return out or None



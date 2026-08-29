"""进度条。不引第三方依赖 —— 这个项目要拷到内网服务器上跑，少一个 pip 装不上
的包就少一处卡壳。

一行原地刷新，非 TTY（重定向到文件、nohup、CI）时自动退化成按百分比打日志行，
不会把日志刷成几万行回车。

    with Progress("VLM 预取", total=90000) as bar:
        for item in work:
            ...
            bar.step(note=f"{ok} 成功 {bad} 失败")
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
from typing import Optional


def _fmt_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:      # 负数或 nan
        return "--"
    if seconds < 90:
        return f"{seconds:.0f}秒"
    if seconds < 5400:
        return f"{seconds / 60:.0f}分"
    return f"{seconds / 3600:.1f}小时"


class Progress:
    """线程安全的进度条 —— 预取是多线程的，step() 会被并发调用。"""

    def __init__(self, label: str, total: int, *, stream=None,
                 min_interval: float = 0.2, log_every_pct: int = 5):
        self.label = label
        self.total = max(0, int(total))
        self.done = 0
        self._stream = stream or sys.stderr
        self._tty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._lock = threading.Lock()
        self._started = time.time()
        self._last_draw = 0.0
        self._min_interval = min_interval
        self._log_every = max(1, log_every_pct)
        self._last_logged_pct = -1
        self._note = ""
        self._closed = False

    # ---------------------------------------------------------------- 使用
    def __enter__(self):
        self._draw(force=True)
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def step(self, n: int = 1, note: str = "") -> None:
        with self._lock:
            self.done += n
            if note:
                self._note = note
            self._draw()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._draw(force=True, final=True)

    # ---------------------------------------------------------------- 绘制
    def _draw(self, force: bool = False, final: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_draw < self._min_interval:
            return
        self._last_draw = now
        elapsed = max(1e-6, now - self._started)
        rate = self.done / elapsed
        pct = (self.done / self.total * 100) if self.total else 0.0
        eta = (self.total - self.done) / rate if rate > 0 and self.total else -1

        if self._tty:
            width = max(20, min(shutil.get_terminal_size((80, 20)).columns, 120))
            body = (f"{self.label} {self.done}/{self.total} "
                    f"({pct:.0f}%) {rate:.1f}/秒 剩余{_fmt_eta(eta)}"
                    + (f"  {self._note}" if self._note else ""))
            barlen = max(0, width - len(body) - 4)
            filled = int(barlen * pct / 100) if barlen else 0
            line = f"{'█' * filled}{'░' * (barlen - filled)} {body}"
            self._stream.write("\r" + line[:width].ljust(width))
            if final:
                self._stream.write(f"\n{self.label} 完成：{self.done}/{self.total}，"
                                   f"用时 {_fmt_eta(elapsed)}\n")
            self._stream.flush()
            return

        # 非 TTY：按百分比打行，不刷屏。重定向到文件时这才是能看的形式。
        step_pct = int(pct // self._log_every) * self._log_every
        if final or step_pct > self._last_logged_pct:
            self._last_logged_pct = step_pct
            self._stream.write(
                f"{self.label} {self.done}/{self.total}（{pct:.0f}%，"
                f"{rate:.1f}/秒，剩余{_fmt_eta(eta)}）"
                + (f" {self._note}" if self._note else "") + "\n")
            self._stream.flush()


class NullProgress:
    """不显示进度时的替身，省掉调用处的 if。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def step(self, n: int = 1, note: str = "") -> None:
        pass

    def close(self) -> None:
        pass


def make(label: str, total: int, enabled: bool = True):
    return Progress(label, total) if enabled and total else NullProgress()

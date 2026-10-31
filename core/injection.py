# -*- coding: utf-8 -*-
"""M4 — 触控注入 (injection)

设计:
  TouchInjector 抽象 ──┬── MinitouchInjector  真机: minitouch socket 协议, 缺依赖优雅报错
                       └── MockInjector       开发: 事件记账 + 按压状态跟踪, 沙箱/CI 全功能可跑

宪法对齐:
  - 弃 adb input tap（30-100ms），只走 minitouch socket（预算 1ms）
  - 注入器只负责"把事件打进去"，不做任何拟人化/调度 —— 那是宏引擎(macro.py)的事
"""
from __future__ import annotations

import shutil
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

from core.constants import FRAME_W, FRAME_H
from core.exceptions import DependencyMissing


@dataclass(frozen=True)
class TouchEvent:
    """单条触控事件。action 取 minitouch 协议字母: d=按下 m=移动 u=抬起。"""
    contact: int          # 触点编号
    x: int                # 1280x720 逻辑坐标（注入器负责映射到真机分辨率）
    y: int
    action: str           # "d" | "m" | "u"
    pressure: float = 1.0

    def __post_init__(self) -> None:
        if self.action not in ("d", "m", "u"):
            raise ValueError(f"非法 action: {self.action!r}（应为 d/m/u）")


@dataclass(frozen=True)
class InjectedEvent:
    """注入器记账条目: 事件 + 注入时刻（clock 口径, 秒）。"""
    event: TouchEvent
    ts: float
    kind: str = "inject"  # "inject" | "reset"


class TouchInjector(ABC):
    """触控注入器接口。inject 返回真实耗时 ms, 供看门狗/丝滑度统计。"""

    def __enter__(self) -> "TouchInjector":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def inject(self, events: list[TouchEvent]) -> float: ...

    def release(self, contact: int) -> float:
        """释放**单个**触点, 不影响其他触点。

        与 reset() 的区别: reset 是"全清"（中断协议兜底）, release 是"精确松手"。
        取消后摇/摇杆持久流收尾必须用 release —— 用 reset 会把正在推着的
        摇杆一起松掉, 走位当场断流。

        默认实现走 inject 单条 'u', 对 minitouch 协议即为 `u <contact>` + `c`。
        """
        return self.inject([TouchEvent(contact, 0, 0, "u")])

    def contacts_pressed(self) -> set[int]:
        """当前按下的触点集合。

        宏引擎靠它决定"哪些触点需要收尾释放"。默认返回空集（表示无法追踪）,
        具体注入器应自行跟踪 —— 不跟踪就不会做精确释放, 退化为依赖 reset 兜底。
        """
        return set()

    @abstractmethod
    def reset(self) -> float:
        """立即释放全部触点（minitouch 'r'）。中断协议的兜底手段。"""


# ═════════════════════════ 开发/测试实现 ═════════════════════════
class MockInjector(TouchInjector):
    """沙箱注入器: 记账所有事件 + 跟踪按压中的触点（中断零残留的验证依据）。

    clock 可注入: 与 VirtualClock 配合时时间戳与调度严格一致,
    与 MonotonicClock 配合时得到真实注入时刻（丝滑度测试用）。
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._log: list[InjectedEvent] = []
        self._pressed: set[int] = set()
        self._lock = __import__("threading").Lock()
        self._opened = False

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        if self._pressed:                     # 关闭前兜底释放
            self.reset()
        self._opened = False

    @property
    def opened(self) -> bool:
        return self._opened

    def inject(self, events: list[TouchEvent]) -> float:
        t0 = time.perf_counter()
        with self._lock:
            now = self._clock()
            for ev in events:
                self._log.append(InjectedEvent(ev, now))
                if ev.action == "d":
                    self._pressed.add(ev.contact)
                elif ev.action == "u":
                    self._pressed.discard(ev.contact)
        return (time.perf_counter() - t0) * 1000.0

    def reset(self) -> float:
        t0 = time.perf_counter()
        with self._lock:
            now = self._clock()
            for c in sorted(self._pressed):
                self._log.append(InjectedEvent(
                    TouchEvent(c, 0, 0, "u"), now, kind="reset"))
            self._pressed.clear()
        return (time.perf_counter() - t0) * 1000.0

    def release(self, contact: int) -> float:
        """精确释放单触点（记账 kind="release", 便于与 reset 区分统计）。"""
        t0 = time.perf_counter()
        with self._lock:
            if contact in self._pressed:
                self._log.append(InjectedEvent(
                    TouchEvent(contact, 0, 0, "u"), self._clock(), kind="release"))
                self._pressed.discard(contact)
        return (time.perf_counter() - t0) * 1000.0

    # ── 观测接口（测试/验收用） ──
    @property
    def events(self) -> list[InjectedEvent]:
        with self._lock:
            return list(self._log)

    def contacts_pressed(self) -> set[int]:
        with self._lock:
            return set(self._pressed)

    def clear_log(self) -> None:
        with self._lock:
            self._log.clear()


# ═════════════════════════ 真机实现 ═════════════════════════
class MinitouchInjector(TouchInjector):
    """minitouch socket 注入（真机）。

    管线: adb forward tcp:<port> → minitouch server socket → 文本协议
          d/m/u <contact> <x> <y> <pressure> + c(commit) / r(reset)
    依赖: adb 二进制 + 设备端已推送 minitouch（任缺 → DependencyMissing）
    分辨率映射: banner 上报 max_x/max_y → 1280x720 逻辑坐标等比缩放。
    """

    def __init__(self, serial: Optional[str] = None, port: int = 11111,
                 host: str = "127.0.0.1", timeout_s: float = 2.0):
        self.serial = serial
        self.port = port
        self.host = host
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None
        self.max_x, self.max_y = FRAME_W, FRAME_H   # banner 覆盖
        self._pressed: set[int] = set()             # 触点跟踪（精确释放的依据）

    def _adb(self, *args: str) -> None:
        cmd = ["adb"] + (["-s", self.serial] if self.serial else []) + list(args)
        import subprocess
        r = subprocess.run(cmd, capture_output=True, timeout=self.timeout_s)
        if r.returncode != 0:
            raise DependencyMissing(
                "adb-device", f"adb {' '.join(args)} 失败: {r.stderr.decode()[:120]}",
                "启动 MuMu 并 adb connect 127.0.0.1:7555",
            )

    def open(self) -> None:
        if shutil.which("adb") is None:
            raise DependencyMissing(
                "adb", "minitouch 注入前置（M4 真机）",
                "安装 Android platform-tools, 并确保模拟器 adb 在线",
            )
        try:
            import subprocess
            # 推送 minitouch（真机联调时提供 binary 路径; 沙箱不会走到这里）
            self._adb("forward", f"tcp:{self.port}", "localabstract:minitouch")
            self._sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout_s)
            self._read_banner()
        except DependencyMissing:
            raise
        except OSError as e:
            raise DependencyMissing(
                "minitouch", f"minitouch socket 连接失败: {e}",
                "设备端推送 minitouch 并运行（abrutus/minitouch）",
            ) from e

    def _read_banner(self) -> None:
        assert self._sock is not None
        f = self._sock.makefile("r")
        meta: dict[str, int] = {}
        for _ in range(6):                    # version/pid/max-contacts/max-x/max-y/max-pressure
            line = f.readline().strip()
            if line.startswith("^"):
                continue
            if line.startswith(" "):
                parts = line.split()
                key = {"0": "contacts", "1": "x", "2": "y", "3": "pressure"}.get(
                    parts[0] if len(parts) > 1 else "", "")
                if key:
                    meta[key] = int(parts[1])
        self.max_x = meta.get("x", FRAME_W)
        self.max_y = meta.get("y", FRAME_H)

    def _scale(self, ev: TouchEvent) -> tuple[int, int]:
        sx = self.max_x / FRAME_W
        sy = self.max_y / FRAME_H
        return (int(round(ev.x * sx)), int(round(ev.y * sy)))

    def inject(self, events: list[TouchEvent]) -> float:
        t0 = time.perf_counter()
        assert self._sock is not None, "MinitouchInjector 未 open()"
        buf: list[str] = []
        for ev in events:
            x, y = self._scale(ev)
            if ev.action == "u":
                buf.append(f"u {ev.contact}\n")
                self._pressed.discard(ev.contact)
            else:
                buf.append(f"{ev.action} {ev.contact} {x} {y} {ev.pressure:g}\n")
                if ev.action == "d":
                    self._pressed.add(ev.contact)
        buf.append("c\n")
        self._sock.sendall("".join(buf).encode())
        return (time.perf_counter() - t0) * 1000.0

    def contacts_pressed(self) -> set[int]:
        return set(self._pressed)

    def reset(self) -> float:
        t0 = time.perf_counter()
        assert self._sock is not None, "MinitouchInjector 未 open()"
        self._sock.sendall(b"r\n")
        self._pressed.clear()
        return (time.perf_counter() - t0) * 1000.0

    def close(self) -> None:
        if self._sock is not None:
            try:
                self.reset()
            except OSError:
                pass
            self._sock.close()
            self._sock = None


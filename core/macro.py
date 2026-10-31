# -*- coding: utf-8 -*-
"""M4 — 宏引擎 (macro)

动作原语:
  Tap      单击（按下 → hold → 抬起）
  Swipe    贝塞尔滑动（三次曲线, 20Hz 采样）
  Joystick 摇杆连续流（20Hz move 流,宪法 JOYSTICK_HZ）
  Wait     纯等待

宏引擎 MacroEngine:
  - 可中断协议: interrupt_event 在每个等待间隙被检查, 触发后立即 reset()
    释放全部触点（中断零残留）——等待用 Event.wait 实现, 响应即时
  - 拟人化（永远开启）: 首事件前 120ms 反应延迟 + 1% 指令级扰动
  - clock 可注入: VirtualClock(单测零等待) / MonotonicClock(真跑)
"""
from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.constants import (
    FRAME_W, FRAME_H,
    HUMANIZE_REACTION_MS, HUMANIZE_MISS_RATE, HUMANIZE_MISS_PX,
    JOYSTICK_HZ,
)
from core.injection import TouchEvent, TouchInjector

JOYSTICK_PERIOD_S = 1.0 / JOYSTICK_HZ          # 50ms


# ═════════════════════════ 时钟抽象 ═════════════════════════
class MonotonicClock:
    """真实时钟。wait() 用 Event.wait 实现 → 可被中断即时唤醒。"""

    def now(self) -> float:
        return time.monotonic()

    def wait(self, seconds: float, stop: Optional[threading.Event]) -> bool:
        """等待 seconds 秒。返回 True 表示因 stop 触发而提前返回。"""
        if stop is None:
            time.sleep(seconds)
            return False
        return stop.wait(seconds)


class VirtualClock:
    """虚拟时钟: wait() 只推进虚拟时间, 单测零真实等待。"""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def wait(self, seconds: float, stop: Optional[threading.Event]) -> bool:
        self.t += seconds
        return bool(stop is not None and stop.is_set())


Clock = Callable  # 结构化类型: 需要 now()/wait()


# ═════════════════════════ 动作原语 ═════════════════════════
# 每个原语产出 [(due_s, TouchEvent), ...], due 为相对宏开始的秒数
Schedule = list[tuple[float, TouchEvent]]


@dataclass(frozen=True)
class Tap:
    cx: int
    cy: int
    contact: int = 0
    hold_ms: float = 40.0

    def schedule(self) -> Schedule:
        ev_d = TouchEvent(self.contact, int(self.cx), int(self.cy), "d")
        ev_u = TouchEvent(self.contact, int(self.cx), int(self.cy), "u")
        return [(0.0, ev_d), (self.hold_ms / 1000.0, ev_u)]


@dataclass(frozen=True)
class Wait:
    ms: float

    def schedule(self) -> Schedule:
        return []                              # 纯等待不产生事件, 引擎读 duration


@dataclass(frozen=True)
class Swipe:
    sx: float
    sy: float
    ex: float
    ey: float
    duration_ms: float
    contact: int = 0
    wobble: float = 0.10                    # 贝塞尔控制点垂直偏移比例

    def schedule(self) -> Schedule:
        pts = self._bezier_points()
        evs: Schedule = []
        n = len(pts)
        for i, (x, y) in enumerate(pts):
            due = self.duration_ms / 1000.0 * i / (n - 1)
            action = "d" if i == 0 else "m"
            evs.append((due, TouchEvent(self.contact, int(round(x)),
                                        int(round(y)), action)))
        evs.append((self.duration_ms / 1000.0,
                    TouchEvent(self.contact, int(round(self.ex)),
                               int(round(self.ey)), "u")))
        return evs

    def _bezier_points(self) -> list[tuple[float, float]]:
        """三次贝塞尔: 控制点取轨迹 30%/70% 处 + 垂直 wobble（拟人手绘弧线）。"""
        dx, dy = self.ex - self.sx, self.ey - self.sy
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length            # 法向
        w = length * self.wobble
        c1 = (self.sx + 0.30 * dx + nx * w, self.sy + 0.30 * dy + ny * w)
        c2 = (self.sx + 0.70 * dx - nx * w, self.sy + 0.70 * dy - ny * w)
        n = max(int(self.duration_ms / 1000.0 / JOYSTICK_PERIOD_S) + 1, 2)
        pts: list[tuple[float, float]] = []
        for i in range(n):
            t = i / (n - 1)
            mt = 1 - t
            x = (mt**3 * self.sx + 3 * mt**2 * t * c1[0]
                 + 3 * mt * t**2 * c2[0] + t**3 * self.ex)
            y = (mt**3 * self.sy + 3 * mt**2 * t * c1[1]
                 + 3 * mt * t**2 * c2[1] + t**3 * self.ey)
            pts.append((x, y))
        return pts


@dataclass(frozen=True)
class Joystick:
    """摇杆连续流: 锚点(keymap joystick) + 方向 → 20Hz move 流直到 duration 结束。"""
    anchor_cx: float
    anchor_cy: float
    anchor_r: float
    dx: float                                   # 期望方向(会归一化)
    dy: float
    duration_ms: float
    strength: float = 1.0                       # 推杆幅度 [0,1]
    contact: int = 0

    def schedule(self) -> Schedule:
        norm = math.hypot(self.dx, self.dy)
        if norm < 1e-6:
            ux, uy = 0.0, 0.0
        else:
            ux, uy = self.dx / norm, self.dy / norm
        ox = ux * self.anchor_r * 0.9 * self.strength
        oy = uy * self.anchor_r * 0.9 * self.strength
        tx, ty = self.anchor_cx + ox, self.anchor_cy + oy
        # 从锚点平滑推出（第一个周期走 30%, 避免瞬间满偏移）
        evs: Schedule = []
        n = max(int(round(self.duration_ms / 1000.0 / JOYSTICK_PERIOD_S)), 1)
        for i in range(n + 1):
            due = i * JOYSTICK_PERIOD_S
            ramp = min(1.0, (i / max(n / 3.0, 1.0)))          # 前 1/3 周期推进
            jx = self.anchor_cx + ox * (0.3 + 0.7 * ramp)
            jy = self.anchor_cy + oy * (0.3 + 0.7 * ramp)
            action = "d" if i == 0 else "m"
            evs.append((due, TouchEvent(self.contact, int(round(jx)),
                                        int(round(jy)), action)))
            tx, ty = jx, jy
        evs.append(((n + 1) * JOYSTICK_PERIOD_S,
                    TouchEvent(self.contact, int(round(tx)),
                               int(round(ty)), "u")))
        return evs


Primitive = Tap | Swipe | Joystick | Wait


def primitive_duration(p: Primitive) -> float:
    """原语 nominal 时长（秒）, 用于排程与 Wait。"""
    if isinstance(p, Wait):
        return p.ms / 1000.0
    if isinstance(p, Tap):
        return p.hold_ms / 1000.0
    if isinstance(p, Swipe):
        return p.duration_ms / 1000.0
    if isinstance(p, Joystick):
        return (len(p.schedule())) * JOYSTICK_PERIOD_S   # 含 down/up 排程
    raise TypeError(f"未知原语: {type(p)}")


@dataclass(frozen=True)
class Macro:
    name: str
    primitives: tuple[Primitive, ...]

    def __post_init__(self) -> None:
        if not self.primitives:
            raise ValueError(f"宏 {self.name!r} 至少需要一个原语")


@dataclass
class MacroResult:
    name: str
    n_events: int
    interrupted: bool = False
    inject_ms: list[float] = field(default_factory=list)   # 每次 inject 耗时
    release_ms: float | None = None                        # 中断 reset 耗时
    start_ts: float | None = None                          # 首事件计划时刻(clock)

    @property
    def inject_p95_ms(self) -> float:
        if not self.inject_ms:
            return 0.0
        xs = sorted(self.inject_ms)
        return xs[min(int(len(xs) * 0.95), len(xs) - 1)]


# ═════════════════════════ 宏引擎 ═════════════════════════
class MacroEngine:
    """把宏排程到注入器。可中断 + 拟人化 + clock 可注入。

    中断协议: stop_event 在每个等待间隙被 wait() 即时唤醒检查;
    触发后立刻 injector.reset() 释放全部触点 → 零残留。
    """

    def __init__(self, injector: TouchInjector,
                 clock: MonotonicClock | VirtualClock | None = None,
                 rng_seed: int | None = 42,
                 humanize: bool = True):
        self.injector = injector
        self.clock = clock if clock is not None else MonotonicClock()
        import random
        self.rng = random.Random(rng_seed)
        self.humanize = humanize

    def _clamp(self, x: float, y: float) -> tuple[int, int]:
        return (int(min(max(round(x), 0), FRAME_W - 1)),
                int(min(max(round(y), 0), FRAME_H - 1)))

    def _build_schedule(self, macro: Macro,
                        t0: float) -> tuple[list[tuple[float, TouchEvent, int]], float]:
        """展开全部原语 → (绝对时刻排程, 总时长)。含拟人化反应延迟与失误扰动。"""
        offset = 0.0
        if self.humanize:
            offset += HUMANIZE_REACTION_MS / 1000.0
        sched: list[tuple[float, TouchEvent, int]] = []
        # 拟人失误: 以 MISS_RATE 概率选一个原语整体偏移 ±MISS_PX
        miss_idx = None
        if self.humanize and self.rng.random() < HUMANIZE_MISS_RATE:
            miss_idx = self.rng.randrange(len(macro.primitives))
        for pi, prim in enumerate(macro.primitives):
            if isinstance(prim, Wait):
                offset += primitive_duration(prim)
                continue
            dx = dy = 0.0
            if pi == miss_idx:
                dx = self.rng.uniform(-HUMANIZE_MISS_PX, HUMANIZE_MISS_PX)
                dy = self.rng.uniform(-HUMANIZE_MISS_PX, HUMANIZE_MISS_PX)
            for due, ev in prim.schedule():
                x, y = self._clamp(ev.x + dx, ev.y + dy)
                sched.append((t0 + offset + due,
                              TouchEvent(ev.contact, x, y, ev.action, ev.pressure),
                              pi))
        offset += primitive_duration(macro.primitives[-1])
        return sched, offset

    def play(self, macro: Macro,
             stop: Optional[threading.Event] = None) -> MacroResult:
        """同步执行宏。stop 触发 → 立即释放全部触点并返回 interrupted=True。"""
        res = MacroResult(name=macro.name, n_events=0)
        t0 = self.clock.now()
        sched, _total = self._build_schedule(macro, t0)
        res.n_events = len(sched)
        if sched:
            res.start_ts = sched[0][0]
        for due, ev, _pi in sched:
            remain = due - self.clock.now()
            if remain > 0 and self.clock.wait(remain, stop):
                res.interrupted = True
                break
            if stop is not None and stop.is_set():
                res.interrupted = True
                break
            res.inject_ms.append(self.injector.inject([ev]))
        # 收尾: 中断 → 强制 reset 兜底（幂等, 保证零残留）; 正常路径按压已随 up 释放
        if res.interrupted or self.injector.contacts_pressed():
            res.release_ms = self.injector.reset()
        return res

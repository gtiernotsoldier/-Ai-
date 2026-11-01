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
    CONTACT_JOYSTICK,
    HUMANIZE_REACTION_MS, HUMANIZE_MISS_RATE, HUMANIZE_MISS_PX,
    JOYSTICK_HZ,
)
from core.human_profile import HumanProfile
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
    # 拟人化分层（宪法「拟人化永远开启」的正确实现方式）:
    #   True  = 反应型动作 —— 首次响应某个新事件, 首事件前加 120ms 反应延迟
    #   False = 序列型动作 —— 已决定好的连段/续流内部事件, 按帧表定时精确触发
    # 真人打连段也不是每一下都重新反应 120ms, 是肌肉记忆一次排好的。
    # 注意: 只有「反应延迟」受此开关控制; 1% 指令级失误扰动始终开启。
    humanize: bool = True

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
                 humanize: bool = True,
                 profile: HumanProfile | None = None):
        self.injector = injector
        self.clock = clock if clock is not None else MonotonicClock()
        import random
        self.rng = random.Random(rng_seed)
        self.humanize = humanize
        # 难度档（Shūkai）: None = 历史默认（120ms / 1%），行为逐字节不变
        self.profile = profile

    @property
    def reaction_ms(self) -> float:
        return self.profile.delay_ms if self.profile is not None else HUMANIZE_REACTION_MS

    @property
    def miss_rate(self) -> float:
        return self.profile.miss_rate if self.profile is not None else HUMANIZE_MISS_RATE

    @property
    def exec_prob(self) -> float:
        return self.profile.exec_prob if self.profile is not None else 1.0

    def set_profile(self, profile: HumanProfile | None) -> None:
        """热切换难度档（同一套权重, 换档即换强度）。"""
        self.profile = profile

    def _clamp(self, x: float, y: float) -> tuple[int, int]:
        return (int(min(max(round(x), 0), FRAME_W - 1)),
                int(min(max(round(y), 0), FRAME_H - 1)))

    def _build_schedule(self, macro: Macro,
                        t0: float) -> tuple[list[tuple[float, TouchEvent, int]], float]:
        """展开全部原语 → (绝对时刻排程, 总时长)。含拟人化反应延迟与失误扰动。

        反应延迟只在 macro.humanize=True（反应型动作）时施加。
        序列型宏（连段内部/续流）必须零延迟 —— 否则 120ms > 50ms 节拍,
        每个宏都会在发出首个事件前就被下一 tick 打断（见 JoystickStream 文档）。
        """
        offset = 0.0
        if self.humanize and macro.humanize:
            # Shūkai 的「动作执行概率」: 走神/手滑, 这条指令根本没发出去。
            # 只在显式降档（exec_prob<1）时抽签 —— 默认档不消耗随机数,
            # 保证历史宏排程逐字节不变（既有测试的种子序列不受影响）。
            if self.exec_prob < 1.0 and self.rng.random() > self.exec_prob:
                return [], 0.0
            offset += self.reaction_ms / 1000.0
        sched: list[tuple[float, TouchEvent, int]] = []
        # 拟人失误: 以 miss_rate 概率选一个原语整体偏移 ±MISS_PX
        miss_idx = None
        if self.humanize and self.rng.random() < self.miss_rate:
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
        # 收尾: 只释放**本宏自己的**触点, 绝不动其他路。
        # 旧实现用全局 reset(), 会把摇杆持久流那一路(CONTACT_JOYSTICK)一起松掉
        # → 分路之后每次按技能都会撕掉走位。全局 reset 只保留给 Executor.release_all()。
        owned = {ev.contact for _due, ev, _pi in sched}
        stuck = owned & self.injector.contacts_pressed()
        if res.interrupted or stuck:
            r_ms = 0.0
            for c in sorted(stuck):
                r_ms = max(r_ms, self.injector.release(c))
            res.release_ms = r_ms        # 标记走过收尾路径（无触点可放时为 0.0）
        return res


# ═════════════════════════ 摇杆持久流 ═════════════════════════
class JoystickStream:
    """长生命周期摇杆流 —— 按下一次, 之后按 JOYSTICK_HZ 持续发 move, 方向在线更新。

    为什么必须有这个类（P0-1 修复）
    ────────────────────────────────
    20Hz 决策每 tick 都会给出**略有不同**的方向（对手一直在动）。旧实现每 tick
    用浮点相等判"同向续流", 方向差一个尾数就中断并重建摇杆宏; 而重建的宏首事件前
    有 HUMANIZE_REACTION_MS=120ms 反应延迟 > DECISION_PERIOD_MS=50ms
    → 每个宏都在发出首个按下事件之前就被下一 tick 打断。

    实测（MockInjector 记账真实触点, 2s / 40 tick）:

        方向完全恒定      → 按下 5 次, move 20 次（每 320ms 断流 120ms, 占空比 62.5%）
        方向每 tick 变 0.05° → 按下 **0** 次, move **0** 次（智能体完全不动）
        方向每 tick 变 1.7°  → 按下 **0** 次, move **0** 次（智能体完全不动）

    本类把「按下一次」与「持续转向」解耦:
      - 按下只发生一次, 且保留 120ms 反应延迟（拟人化: 这属于反应型动作）
      - 转向是零成本在线更新: 不释放触点、不重建宏、不加延迟
      - 方向变化时允许提前下发（下限 STEER_MIN_INTERVAL_S）, 把最坏转向延迟
        从 50ms 压到 20ms —— 走位躲技能靠的就是这 30ms
      - 收尾用 injector.release(contact) 精确松手, 不影响按钮那一路触点
    """

    STEER_MIN_INTERVAL_S = 0.02      # 转向提前下发的最小间隔（限流 ≤50Hz）
    RAMP_PERIODS = 3.0               # 推杆从 30% 平滑推到 100% 所需周期数

    def __init__(self, engine: "MacroEngine", cx: float, cy: float, r: float,
                 contact: int = CONTACT_JOYSTICK):
        self.engine = engine
        self.cx, self.cy, self.r = float(cx), float(cy), float(r)
        self.contact = contact
        self._lock = threading.Lock()
        self._dir: tuple[float, float] = (0.0, 0.0)
        self._strength: float = 1.0
        self._stop = threading.Event()
        self._wake = threading.Event()      # set_direction 转向 或 stop 都会唤醒
        self._thread: threading.Thread | None = None
        self._pressed = False               # 触点当前是否按下
        self._press_t: float = 0.0
        self.inject_ms: list[float] = []
        self.n_events: int = 0
        self.n_steers: int = 0              # 提前下发（非节拍点）的次数

    # ── 方向在线更新（决策线程调用, 零释放零重建）──
    def set_direction(self, dx: float, dy: float, strength: float = 1.0) -> None:
        with self._lock:
            self._dir = (float(dx), float(dy))
            self._strength = float(min(max(strength, 0.0), 1.0))
        self._wake.set()                    # 唤醒流线程立即评估是否提前下发

    def _target(self, now: float) -> tuple[int, int]:
        with self._lock:
            dx, dy = self._dir
            s = self._strength
        n = math.hypot(dx, dy)
        if n < 1e-6:
            ox = oy = 0.0                   # 回中: 停走但手指仍握杆
        else:
            ox = dx / n * self.r * 0.9 * s
            oy = dy / n * self.r * 0.9 * s
        ramp = min(1.0, (now - self._press_t) / (self.RAMP_PERIODS * JOYSTICK_PERIOD_S)) \
            if self._press_t else 1.0
        f = 0.3 + 0.7 * ramp
        return (int(round(self.cx + ox * f)), int(round(self.cy + oy * f)))

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def pressed(self) -> bool:
        return self._pressed

    def start(self) -> None:
        """启动流线程。幂等: 已存活则直接返回。"""
        if self.alive:
            return
        self._stop.clear()
        self._wake.clear()
        self._thread = threading.Thread(target=self._run,
                                        name="joystick-stream", daemon=True)
        self._thread.start()

    def _emit(self, x: int, y: int, action: str) -> None:
        self.inject_ms.append(
            self.engine.injector.inject([TouchEvent(self.contact, x, y, action)]))
        self.n_events += 1

    def _run(self) -> None:
        clock = self.engine.clock
        # 首次按下属"反应型动作" → 保留拟人化反应延迟（宪法: 拟人化永远开启）
        if self.engine.humanize:
            if clock.wait(self.engine.reaction_ms / 1000.0, self._stop):
                return
        self._press_t = clock.now()
        x, y = self._target(self._press_t)
        self._emit(x, y, "d")
        self._pressed = True
        try:
            last_emit = self._press_t
            while not self._stop.is_set():
                remain = JOYSTICK_PERIOD_S - (clock.now() - last_emit)
                if remain > 0:
                    self._wake.clear()
                    clock.wait(remain, self._wake)      # 被转向或 stop 唤醒
                if self._stop.is_set():
                    break
                now = clock.now()
                early = (now - last_emit) < JOYSTICK_PERIOD_S * 0.9
                if early and (now - last_emit) < self.STEER_MIN_INTERVAL_S:
                    continue                            # 限流: 距上次下发太近
                nx, ny = self._target(now)
                if early and (nx, ny) == (x, y):
                    continue                            # 方向没变, 不必提前下发
                if early:
                    self.n_steers += 1
                self._emit(nx, ny, "m")
                x, y = nx, ny
                last_emit = now
        finally:
            # 精确松手: 只释放摇杆这一路, 按钮触点(CONTACT_BUTTON)不受影响
            if self._pressed:
                self.engine.injector.release(self.contact)
                self._pressed = False

    def stop(self, timeout: float = 0.1) -> None:
        """停流并释放触点。幂等。"""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

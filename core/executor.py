# -*- coding: utf-8 -*-
"""M4 — 执行层门面 (executor)

职责:
  - 绑定 keymap 标定 → 语义化动作: tap_button("ult") / joystick(dx,dy,ms) / swipe(...)
  - 拼装宏 → MacroEngine 同步/异步执行（play_async 返回可中断句柄）
  - smoothness_test [B] 项的验收实现: 摇杆无停顿帧 / 注入<20ms / 中断零残留
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from core.constants import (
    EXEC_INJECT_LIMIT_MS, GAP_TOLERANCE_X, JOYSTICK_HZ,
)
from core.injection import MockInjector, TouchInjector
from core.keymap import Keymap
from core.macro import (
    Macro, MacroEngine, MacroResult, MonotonicClock, VirtualClock,
    Joystick, Swipe, Tap,
)

JOYSTICK_PERIOD_MS = 1000.0 / JOYSTICK_HZ


class MacroHandle:
    """异步宏句柄: interrupt() 即时唤醒引擎等待循环。"""

    def __init__(self, thread: threading.Thread, stop: threading.Event,
                 getter: Callable[[], Optional[MacroResult]]):
        self._thread = thread
        self._stop = stop
        self._getter = getter

    def interrupt(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    @property
    def result(self) -> Optional[MacroResult]:
        return self._getter()

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()


class Executor:
    """执行层门面。injector 可换 Mock/真机 minitouch, clock 可注入。"""

    def __init__(self, injector: TouchInjector, keymap: Keymap,
                 clock: MonotonicClock | VirtualClock | None = None,
                 rng_seed: int | None = 42, humanize: bool = True):
        self.km = keymap
        self.injector = injector
        self.engine = MacroEngine(injector, clock=clock,
                                  rng_seed=rng_seed, humanize=humanize)

    # ── 宏拼装（语义 → 原语） ──
    def tap_macro(self, button: str, hold_ms: float = 40.0) -> Macro:
        cx, cy, _r = self.km.circle(f"btn_{button}")
        return Macro(f"tap_{button}", (Tap(cx, cy, hold_ms=hold_ms),))

    def joystick_macro(self, dx: float, dy: float, duration_ms: float,
                       strength: float = 1.0) -> Macro:
        cx, cy, r = self.km.circle("joystick")
        return Macro("joystick", (Joystick(cx, cy, r, dx, dy,
                                           duration_ms, strength=strength),))

    def swipe_macro(self, sx: float, sy: float, ex: float, ey: float,
                    duration_ms: float) -> Macro:
        return Macro("swipe", (Swipe(sx, sy, ex, ey, duration_ms),))

    def combo_macro(self, primitives: tuple, name: str = "combo") -> Macro:
        return Macro(name, tuple(primitives))

    # ── 执行 ──
    def play(self, macro: Macro,
             stop: Optional[threading.Event] = None) -> MacroResult:
        return self.engine.play(macro, stop=stop)

    def play_async(self, macro: Macro) -> MacroHandle:
        stop = threading.Event()
        box: dict[str, MacroResult | None] = {"res": None}

        def _run() -> None:
            box["res"] = self.engine.play(macro, stop=stop)

        th = threading.Thread(target=_run, name=f"macro-{macro.name}", daemon=True)
        th.start()
        return MacroHandle(th, stop, lambda: box["res"])

    def release_all(self) -> None:
        self.injector.reset()


# ═════════════════════════ 丝滑度 [B] 验收 ═════════════════════════
def check_executor() -> tuple[bool, str]:
    """[B] 摇杆轨迹无停顿帧 / 注入<20ms / 中断零残留。

    用 MockInjector + MonotonicClock 真跑 1s 摇杆流, 出真实数字。
    """
    import numpy as np

    km = Keymap.load()
    inj = MockInjector(clock=time.monotonic)
    ex = Executor(inj, km, humanize=True)          # 拟人永远开启
    ok = True

    # ── B1: 摇杆 20Hz 连续流, 相邻事件间隔无停顿帧 ──
    res = ex.play(ex.joystick_macro(0.6, -0.8, duration_ms=1000.0))
    evs = [e for e in inj.events if e.kind == "inject"]
    ts = np.array([e.ts for e in evs])
    gaps = np.diff(ts) * 1000.0
    max_gap = float(gaps.max()) if len(gaps) else 0.0
    p95_gap = float(np.percentile(gaps, 95)) if len(gaps) else 0.0
    limit = JOYSTICK_PERIOD_MS * GAP_TOLERANCE_X
    b1 = len(evs) >= 18 and max_gap <= limit and not res.interrupted
    ok &= b1

    # ── B2: 单次注入耗时 p95 < 20ms（看门狗口径） ──
    inj_p95 = res.inject_p95_ms
    b2 = inj_p95 <= EXEC_INJECT_LIMIT_MS
    ok &= b2

    # ── B3: 中断即时唤醒 + 释放全部触点（零残留） ──
    inj.clear_log()
    h = ex.play_async(ex.joystick_macro(1.0, 0.0, duration_ms=3000.0))
    time.sleep(0.3)
    t_int = time.monotonic()
    h.interrupt()
    h.join(timeout=2.0)
    interrupt_ms = (time.monotonic() - t_int) * 1000.0
    r = h.result
    residual = inj.contacts_pressed()
    b3 = (r is not None and r.interrupted and not residual
          and interrupt_ms <= EXEC_INJECT_LIMIT_MS)
    ok &= b3

    msg = (f"[B] 摇杆流: {len(evs)} 事件, 间隔 p95={p95_gap:.2f}ms max={max_gap:.2f}ms "
           f"(容忍{limit:.0f}ms) {'PASS' if b1 else 'FAIL'} | "
           f"注入 p95={inj_p95:.3f}ms (<{EXEC_INJECT_LIMIT_MS:.0f}ms) "
           f"{'PASS' if b2 else 'FAIL'} | "
           f"中断响应={interrupt_ms:.2f}ms, 残留触点={len(residual)} "
           f"{'PASS' if b3 else 'FAIL'}")
    return ok, msg

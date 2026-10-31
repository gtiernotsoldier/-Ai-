# -*- coding: utf-8 -*-
"""M4.5 — 20Hz 闭环主循环 (orchestrator)

节拍（宪法 DECISION_HZ=20）:
  捕获 → 感知(快60Hz/慢12Hz) → 状态组装 → 决策(看门狗20ms沿用上帧) → 下发执行
  执行走 play_async 可中断宏协议: 本 tick 50ms 动作流与下 tick 感知重叠,
  新决策到来即中断旧流（M4 中断零残留协议）。

闭环纪律: 主循环内零网络/零磁盘/零网络注入; pacing sleep 仅是实时节拍器
（等价 smoothness_test 的离线 sleep 口径）。

python -m core.orchestrator  →  沙箱全链路自走演示 + 闭环延迟报告
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from core.constants import (
    DECISION_PERIOD_MS, DECISION_TIMEOUT_MS, LOOP_BUDGET_MS,
    REDLINE_LATENCY_MS, TARGET_LATENCY_MS,
)
from core.executor import Executor, MacroHandle
from core.injection import MockInjector, TouchInjector
from core.keymap import Keymap
from core.macro import Macro, MonotonicClock, VirtualClock, Wait
from core.perception import Perception
from core.policy import Action, PolicyProvider
from core.screen_capture import FrameSource, SyntheticSource
from core.state import StateBuilder
from core.synthetic import Scene


# ═════════════════════════ 闭环报告 ═════════════════════════
@dataclass
class LoopReport:
    ticks: int = 0
    actual_hz: float = 0.0
    watchdog_hits: int = 0
    action_counts: dict = field(default_factory=dict)
    # 各阶段耗时(ms)样本
    capture_ms: list = field(default_factory=list)
    perceive_ms: list = field(default_factory=list)
    state_ms: list = field(default_factory=list)
    decide_ms: list = field(default_factory=list)
    issue_ms: list = field(default_factory=list)
    loop_ms: list = field(default_factory=list)     # 帧就绪(frame.ts) → 下发完成
    full_ms: list = field(default_factory=list)     # 捕获起点 → 下发完成

    def _p(self, xs: list, q: float) -> float:
        return float(np.percentile(xs, q)) if xs else 0.0

    def render(self) -> str:
        rows = [
            ("画面捕获", self.capture_ms, 10.0),
            ("感知推理", self.perceive_ms, 15.0),
            ("状态组装", self.state_ms, 1.0),
            ("决策(看门狗20ms)", self.decide_ms, 15.0),
            ("动作下发", self.issue_ms, 1.0),
            ("感知→下发(帧就绪起)", self.loop_ms, 32.0),
            ("闭环合计(捕获起)", self.full_ms, LOOP_BUDGET_MS),
        ]
        lines = [f"ticks={self.ticks}  实测节拍={self.actual_hz:.2f}Hz  "
                 f"看门狗触发={self.watchdog_hits}次",
                 "动作分布: " + (", ".join(f"{k}×{v}" for k, v in
                                     sorted(self.action_counts.items())) or "-"),
                 f"{'环节':<22}{'预算':>7}{'p50':>9}{'p95':>9}{'max':>9}  判定"]
        ok_all = True
        for name, xs, budget in rows:
            p50, p95, mx = self._p(xs, 50), self._p(xs, 95), max(xs) if xs else 0.0
            ok = p95 <= budget
            ok_all &= ok
            lines.append(f"{name:<22}{budget:>7.1f}{p50:>9.2f}{p95:>9.2f}"
                         f"{mx:>9.2f}  {'PASS' if ok else 'FAIL'}")
        pct_red = self._p(self.full_ms, 95) / REDLINE_LATENCY_MS * 100
        pct_tgt = self._p(self.full_ms, 95) / TARGET_LATENCY_MS * 100
        lines.append(f"闭环 p95 占红线 {pct_red:.1f}% / 目标 {pct_tgt:.1f}% "
                     f"(不含模拟器渲染16ms+响应16ms 物理税)")
        return "\n".join(lines)


# ═════════════════════════ 主循环 ═════════════════════════
class Orchestrator:
    """20Hz 感知→决策→执行闭环。"""

    def __init__(self, source: FrameSource, perception: Perception,
                 builder: StateBuilder, policy: PolicyProvider,
                 executor: Executor,
                 clock: MonotonicClock | VirtualClock | None = None):
        self.source = source
        self.perception = perception
        self.builder = builder
        self.policy = policy
        self.executor = executor
        self.clock = clock if clock is not None else MonotonicClock()
        self._stop = threading.Event()
        self._handle: MacroHandle | None = None

    # ── 动作 → 宏 ──
    def _macro_of(self, action: Action) -> Macro:
        if action.button is not None:
            return self.executor.tap_macro(action.button, hold_ms=40.0)
        dx, dy = action.move
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            return self.executor.joystick_macro(dx, dy, action.duration_ms)
        return Macro("idle", (Wait(action.duration_ms),))

    def _issue(self, action: Action) -> float:
        """同向动作流延续; 变更时中断旧流(join 等待释放)再下发新动作。"""
        if (self._last_action == action and self._handle is not None
                and not self._handle.done):
            return 0.0                            # 摇杆流自行 20Hz 续流
        t0 = time.perf_counter()
        if self._handle is not None and not self._handle.done:
            self._handle.interrupt()              # 中断协议: 即时唤醒+零残留
            self._handle.join(timeout=0.05)       # 保证 reset 完成后再按下
        self._handle = self.executor.play_async(self._macro_of(action))
        return (time.perf_counter() - t0) * 1000.0

    def stop(self) -> None:
        self._stop.set()

    def run(self, ticks: int | None = None,
            duration_s: float | None = None) -> LoopReport:
        """跑闭环: 指定 tick 数或时长(秒), 二者均缺省=跑 1 秒。"""
        rpt = LoopReport()
        period = DECISION_PERIOD_MS / 1000.0
        t_start = self.clock.now()
        t_deadline = (t_start + duration_s) if duration_s else None
        last_action: Action | None = None
        self._last_action: Action | None = None
        n = 0
        try:
            with self.source:
                while not self._stop.is_set():
                    if ticks is not None and n >= ticks:
                        break
                    if t_deadline is not None and self.clock.now() >= t_deadline:
                        break
                    t_tick = self.clock.now()

                    # ── 感知链路 ──
                    t0 = time.perf_counter()
                    frame = self.source.read()
                    rpt.capture_ms.append((time.perf_counter() - t0) * 1000)
                    t_capture0 = t0                     # 捕获起点(含渲染)
                    t0 = time.perf_counter()
                    res = self.perception.update(frame)
                    rpt.perceive_ms.append((time.perf_counter() - t0) * 1000)
                    t0 = time.perf_counter()
                    state = self.builder.build(res, frame.frame_id, frame.ts)
                    rpt.state_ms.append((time.perf_counter() - t0) * 1000)

                    # ── 决策 + 看门狗: 超时沿用上帧动作 ──
                    t0 = time.perf_counter()
                    action = self.policy.decide(state)
                    decide_ms = (time.perf_counter() - t0) * 1000
                    rpt.decide_ms.append(decide_ms)
                    if decide_ms > DECISION_TIMEOUT_MS:
                        rpt.watchdog_hits += 1
                        action = last_action if last_action is not None else action

                    # ── 下发 ──
                    rpt.issue_ms.append(self._issue(action))
                    key = action.button if action.button is not None else "move"
                    rpt.action_counts[key] = rpt.action_counts.get(key, 0) + 1
                    last_action = action
                    self._last_action = action
                    _now = time.perf_counter()
                    rpt.loop_ms.append(_now * 1000 - frame.ts * 1000)
                    rpt.full_ms.append((_now - t_capture0) * 1000)
                    n += 1
                    rpt.ticks = n

                    # ── 实时节拍器 ──
                    remain = period - (self.clock.now() - t_tick)
                    if remain > 0:
                        self.clock.wait(remain, self._stop)
        finally:
            # 收尾: 中断在飞动作流, 保证零残留
            if self._handle is not None:
                self._handle.interrupt()
                self._handle.join(timeout=1.0)
        elapsed = self.clock.now() - t_start
        rpt.actual_hz = rpt.ticks / elapsed if elapsed > 0 else 0.0
        return rpt


# ═════════════════════════ 沙箱自走演示 ═════════════════════════
def main(duration_s: float = 3.0) -> int:
    print("=" * 78)
    print("Shinobi_God 闭环演示 (M1-M4.5 全链路: 捕获→感知→状态→决策→执行)")
    print("=" * 78)
    src = SyntheticSource(fps=60, scene=Scene())
    from core.detectors import MockFastDetector
    from core.perception import CVReader
    perc = Perception(MockFastDetector(), CVReader(Keymap.load()), slow_every=5)
    builder = StateBuilder()
    from core.policy import RulePolicy
    policy = RulePolicy()
    inj = MockInjector(clock=time.monotonic)
    executor = Executor(inj, Keymap.load(), humanize=True)
    orch = Orchestrator(src, perc, builder, policy, executor)

    rpt = orch.run(duration_s=duration_s)
    print()
    print(rpt.render())
    residual = inj.contacts_pressed()
    print(f"残留触点: {len(residual)} → {'PASS' if not residual else 'FAIL'}")
    ok = (rpt.ticks >= duration_s * 15 and not residual
          and rpt._p(rpt.full_ms, 95) <= LOOP_BUDGET_MS)
    print("\n纪律提醒: 闭环内零网络/零磁盘; pacing sleep 仅是实时节拍器。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

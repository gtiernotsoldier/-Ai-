# -*- coding: utf-8 -*-
"""M4.5 单测 — 20Hz 主循环 (orchestrator): 节拍 / 看门狗 / 零残留 / 动作下发。"""
from __future__ import annotations

import time

import numpy as np
import pytest

from core.constants import DECISION_TIMEOUT_MS, LOOP_BUDGET_MS
from core.detectors import MockFastDetector
from core.executor import Executor
from core.injection import MockInjector
from core.keymap import Keymap
from core.orchestrator import Orchestrator
from core.perception import CVReader, Perception
from core.policy import Action, PolicyProvider, RulePolicy
from core.screen_capture import SyntheticSource
from core.state import StateBuilder
from core.synthetic import Scene


class FixedPolicy(PolicyProvider):
    """固定动作策略: 验证动作→宏映射与动作计数。"""
    name = "fixed"

    def __init__(self, action: Action):
        self.action = action

    def decide(self, state: np.ndarray) -> Action:
        return self.action


def make_orch(policy, humanize=False):
    src = SyntheticSource(fps=60, scene=Scene())
    km = Keymap.load()
    perc = Perception(MockFastDetector(), CVReader(km), slow_every=5)
    inj = MockInjector(clock=time.monotonic)
    ex = Executor(inj, km, humanize=humanize)
    return Orchestrator(src, perc, StateBuilder(), policy, ex), inj


class TestLoop:
    def test_runs_real_time_20hz(self):
        orch, inj = make_orch(RulePolicy())
        t0 = time.perf_counter()
        rpt = orch.run(ticks=20)
        wall = time.perf_counter() - t0
        assert rpt.ticks == 20
        assert 0.8 <= wall <= 2.0                 # 20 ticks @ 50ms ≈ 1s
        assert rpt.actual_hz >= 15.0              # 实测节拍接近 20Hz
        assert rpt.watchdog_hits == 0             # 规则策略不会超时

    def test_action_counts_recorded(self):
        orch, inj = make_orch(FixedPolicy(Action(button="ult")))
        rpt = orch.run(ticks=5)
        assert rpt.action_counts.get("ult") == 5

    def test_zero_residual_after_run(self):
        orch, inj = make_orch(RulePolicy())
        orch.run(ticks=20)
        assert inj.contacts_pressed() == set()    # 收尾中断+兜底释放

    def test_loop_latency_within_budget(self):
        orch, inj = make_orch(RulePolicy())
        rpt = orch.run(ticks=30)
        p95 = float(np.percentile(rpt.full_ms, 95))
        assert p95 <= LOOP_BUDGET_MS              # 捕获起点→下发 预算 42ms

    def test_joystick_action_produces_events(self):
        orch, inj = make_orch(FixedPolicy(Action(move=(1.0, 0.0))))
        rpt = orch.run(ticks=6)                   # 6 ticks=300ms > 200ms 流窗口
        evs = [e for e in inj.events if e.kind == "inject"]
        # 同向动作由主循环自动续流: 一次完整 200ms 流 = d + 4×m (+u)
        assert len(evs) >= 6
        assert any(e.event.action == "d" for e in evs)
        assert rpt.action_counts.get("move") == 6

    def test_tap_action_hits_button_center(self):
        orch, inj = make_orch(FixedPolicy(Action(button="skill2")))
        orch.run(ticks=2)
        km = Keymap.load()
        cx, cy, _ = km.circle("btn_skill2")
        downs = [e.event for e in inj.events
                 if e.kind == "inject" and e.event.action == "d"]
        assert downs and (downs[0].x, downs[0].y) == (cx, cy)


class TestWatchdog:
    def test_slow_policy_triggers_watchdog_and_reuses_last(self):
        """前向 >20ms → 看门狗计数 + 沿用上帧动作(动作计数仍连续)。"""
        from core.policy import SlowPolicy
        orch, inj = make_orch(SlowPolicy(FixedPolicy(Action(move=(0.0, -1.0))),
                                         delay_s=0.03))
        rpt = orch.run(ticks=4)
        assert rpt.watchdog_hits == 4             # 每次 30ms 前向都超时
        assert rpt.action_counts.get("move") == 4  # 沿用上帧 → 仍下发 move
        assert inj.contacts_pressed() == set()


class TestStop:
    def test_stop_event_breaks_run(self):
        orch, inj = make_orch(RulePolicy())
        import threading
        threading.Timer(0.2, orch.stop).start()
        rpt = orch.run(duration_s=10.0)           # 0.2s 后被 stop 打断
        assert rpt.ticks < 200
        assert inj.contacts_pressed() == set()

# -*- coding: utf-8 -*-
"""M4 单测 — 宏引擎 (macro): 原语排程 / 拟人化 / 可中断协议 / 零残留。"""
from __future__ import annotations

import math
import threading
import time

import pytest

import core.macro as macro_mod
from core.injection import MockInjector
from core.macro import (
    JOYSTICK_PERIOD_S,
    Joystick, Macro, MacroEngine, MonotonicClock, Swipe, Tap, VirtualClock,
    Wait, primitive_duration,
)


def make_engine(inj=None, humanize=False, seed=7):
    clock = VirtualClock()
    inj = inj or MockInjector(clock=clock.now)
    return MacroEngine(inj, clock=clock, rng_seed=seed, humanize=humanize), inj, clock


class TestPrimitives:
    def test_tap_schedule_down_then_up(self):
        s = Tap(100, 200, hold_ms=40).schedule()
        assert [ev.action for _, ev in s] == ["d", "u"]
        assert s[0][0] == 0.0 and s[1][0] == pytest.approx(0.04)
        assert (s[0][1].x, s[0][1].y) == (100, 200)

    def test_swipe_bezier_monotonic_and_endpoints(self):
        sw = Swipe(200, 600, 900, 300, duration_ms=400)
        s = sw.schedule()
        dues = [d for d, _ in s]
        assert dues == sorted(dues)
        assert s[0][1].action == "d" and s[-1][1].action == "u"
        assert abs(s[-1][1].x - 900) <= 1 and abs(s[-1][1].y - 300) <= 1
        # 20Hz 采样: 400ms → ~9 个事件
        assert 7 <= len(s) <= 11

    def test_joystick_flow_20hz_and_bounds(self):
        j = Joystick(180, 600, 70, 1.0, 0.0, duration_ms=1000)
        s = j.schedule()
        assert s[0][1].action == "d" and s[-1][1].action == "u"
        # 事件数 ≈ duration/period
        assert len(s) == pytest.approx(1000 / 50.0 + 2, abs=2)
        # 相邻 due 严格 50ms（无停顿帧的排程保证）
        dues = [d for d, _ in s[1:-1]]
        for a, b in zip(dues, dues[1:]):
            assert b - a == pytest.approx(JOYSTICK_PERIOD_S, abs=1e-9)
        # 轨迹点不超出锚点半径
        for _, ev in s:
            d = math.hypot(ev.x - 180, ev.y - 600)
            assert d <= 70 * 0.95 + 2

    def test_joystick_zero_direction_hold_center(self):
        j = Joystick(180, 600, 70, 0.0, 0.0, duration_ms=200)
        s = j.schedule()
        assert all(abs(ev.x - 180) <= 2 and abs(ev.y - 600) <= 2 for _, ev in s)

    def test_wait_no_events(self):
        assert Wait(150).schedule() == []
        assert primitive_duration(Wait(150)) == pytest.approx(0.15)


class TestEngine:
    def test_play_produces_events_in_order(self):
        eng, inj, _ = make_engine()
        res = eng.play(Macro("t", (Tap(100, 200),)))
        assert not res.interrupted
        evs = [e.event for e in inj.events]
        assert [e.action for e in evs] == ["d", "u"]
        assert res.n_events == 2 and len(res.inject_ms) == 2

    def test_virtual_clock_zero_wait(self):
        eng, inj, _ = make_engine()
        t0 = time.perf_counter()
        eng.play(Macro("js", (Joystick(180, 600, 70, 0.5, 0.5, 1000),)))
        assert time.perf_counter() - t0 < 0.2        # 1s 宏零真实等待

    def test_combo_with_wait_offset(self):
        eng, inj, _ = make_engine()
        res = eng.play(Macro("c", (Wait(200), Tap(100, 100))))
        first = inj.events[0]
        assert first.ts >= 0.2                       # Wait 把 Tap 推后 200ms
        assert res.n_events == 2

    def test_humanize_reaction_delay(self):
        eng, inj, _ = make_engine(humanize=True)
        t0 = eng.clock.now()
        res = eng.play(Macro("t", (Tap(100, 100),)))
        assert res.start_ts - t0 >= 0.119            # 120ms 反应延迟

    def test_humanize_no_delay_when_off(self):
        eng, inj, _ = make_engine(humanize=False)
        t0 = eng.clock.now()
        res = eng.play(Macro("t", (Tap(100, 100),)))
        assert res.start_ts == pytest.approx(t0, abs=1e-9)

    def test_humanize_miss_injection(self, monkeypatch):
        """失误率=1 时必有一次原语级偏移。"""
        monkeypatch.setattr(macro_mod, "HUMANIZE_MISS_RATE", 1.0)
        monkeypatch.setattr(macro_mod, "HUMANIZE_MISS_PX", 12.0)
        eng, inj, _ = make_engine(humanize=True, seed=7)
        eng.play(Macro("t", (Tap(100, 100, hold_ms=50),)))
        evs = [e.event for e in inj.events]
        assert any(abs(e.x - 100) > 0.5 or abs(e.y - 100) > 0.5 for e in evs)

    def test_coordinates_clamped_to_frame(self, monkeypatch):
        monkeypatch.setattr(macro_mod, "HUMANIZE_MISS_RATE", 1.0)
        monkeypatch.setattr(macro_mod, "HUMANIZE_MISS_PX", 500.0)  # 越界扰动
        eng, inj, _ = make_engine(humanize=True, seed=3)
        eng.play(Macro("t", (Tap(5, 5, hold_ms=20),)))
        for e in inj.events:
            assert 0 <= e.event.x <= 1279 and 0 <= e.event.y <= 719


class TestInterrupt:
    def test_stop_before_play_immediate_interrupt(self):
        eng, inj, _ = make_engine()
        stop = threading.Event()
        stop.set()
        res = eng.play(Macro("js", (Joystick(180, 600, 70, 1, 0, 1000),)), stop=stop)
        assert res.interrupted
        assert inj.contacts_pressed() == set()       # 零残留
        assert res.release_ms is not None            # 走了 reset 兜底

    def test_interrupt_mid_play_real_clock(self):
        """真实时钟: 300ms 宏在 ~100ms 被打断 → 零残留 + 及时退出。"""
        clock = MonotonicClock()
        inj = MockInjector(clock=clock.now)
        eng = MacroEngine(inj, clock=clock, humanize=False)
        stop = threading.Event()

        def killer():
            time.sleep(0.1)
            stop.set()

        threading.Thread(target=killer, daemon=True).start()
        t0 = time.monotonic()
        res = eng.play(Macro("js", (Joystick(180, 600, 70, 1, 0, 300),)), stop=stop)
        wall = time.monotonic() - t0
        assert res.interrupted
        assert wall < 0.25                           # 没跑满 300ms + 反应延迟
        assert inj.contacts_pressed() == set()       # 中断零残留

    def test_normal_play_zero_residual(self):
        eng, inj, _ = make_engine()
        eng.play(Macro("js", (Joystick(180, 600, 70, 0.3, 0.4, 300),)))
        assert inj.contacts_pressed() == set()       # 正常结束也应无残留

    def test_interrupt_p95_inject_recorded(self):
        eng, inj, _ = make_engine()
        stop = threading.Event()
        stop.set()
        res = eng.play(Macro("js", (Joystick(180, 600, 70, 1, 0, 1000),)), stop=stop)
        assert res.inject_p95_ms >= 0.0

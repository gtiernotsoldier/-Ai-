# -*- coding: utf-8 -*-
"""M4 单测 — 执行层门面 (executor): 语义动作 / 异步中断 / 验收检查。"""
from __future__ import annotations

import threading
import time

import pytest

from core.executor import Executor, check_executor
from core.injection import MockInjector
from core.keymap import Keymap
from core.macro import VirtualClock


@pytest.fixture()
def km():
    return Keymap.load()


@pytest.fixture()
def ex(km):
    clock = VirtualClock()
    inj = MockInjector(clock=clock.now)
    return Executor(inj, km, clock=clock, humanize=False), inj


class TestFacade:
    def test_tap_macro_uses_keymap_center(self, ex):
        ex, inj = ex
        km = ex.km
        cx, cy, _ = km.circle("btn_ult")
        ex.play(ex.tap_macro("ult", hold_ms=30))
        first = inj.events[0].event
        assert (first.x, first.y) == (cx, cy)

    def test_joystick_macro_uses_keymap_anchor(self, ex):
        ex, inj = ex
        ex.play(ex.joystick_macro(1.0, 0.0, duration_ms=150))
        evs = inj.events
        assert evs[0].event.action == "d"
        assert evs[-1].event.action == "u"
        # 推杆点在摇杆锚点右侧
        assert evs[2].event.x > 180

    def test_swipe_macro_endpoints(self, ex):
        ex, inj = ex
        ex.play(ex.swipe_macro(100, 100, 500, 400, duration_ms=200))
        evs = [e.event for e in inj.events]
        assert evs[0].action == "d" and evs[-1].action == "u"
        assert abs(evs[-1].x - 500) <= 1 and abs(evs[-1].y - 400) <= 1

    def test_combo_macro(self, ex):
        ex, inj = ex
        from core.macro import Tap, Wait
        m = ex.combo_macro((Tap(200, 300), Wait(50), Tap(400, 500)), name="连招")
        res = ex.play(m)
        assert res.n_events == 4                     # 2 taps = d/u each

    def test_release_all(self, ex):
        ex, inj = ex
        stop = threading.Event()
        stop.set()
        ex.play(ex.joystick_macro(1, 0, 500), stop=stop)
        ex.release_all()
        assert inj.contacts_pressed() == set()


class TestAsyncInterrupt:
    def test_play_async_interrupt_latency_and_residual(self, km):
        """真实时钟: 3s 摇杆流 150ms 时打断 → <20ms 响应 + 零残留。"""
        inj = MockInjector(clock=time.monotonic)
        ex = Executor(inj, km, humanize=False)
        h = ex.play_async(ex.joystick_macro(1.0, 0.0, duration_ms=3000))
        time.sleep(0.15)
        t0 = time.monotonic()
        h.interrupt()
        h.join(timeout=1.0)
        lat = (time.monotonic() - t0) * 1000
        assert h.done and h.result is not None
        assert h.result.interrupted
        assert inj.contacts_pressed() == set()
        assert lat < 20.0

    def test_play_async_completes_normally(self, km):
        clock = VirtualClock()
        inj = MockInjector(clock=clock.now)
        ex = Executor(inj, km, clock=clock, humanize=False)
        h = ex.play_async(ex.tap_macro("skill1"))
        h.join(timeout=1.0)
        assert h.done
        r = h.result
        assert r is not None and not r.interrupted and r.n_events == 2


class TestAcceptance:
    def test_check_executor_pass(self):
        """[B] 丝滑度验收全项: 摇杆无停顿帧 / 注入<20ms / 中断零残留。"""
        ok, msg = check_executor()
        assert ok, msg
        assert "PASS" in msg

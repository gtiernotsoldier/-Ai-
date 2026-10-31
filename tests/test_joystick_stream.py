# -*- coding: utf-8 -*-
"""P0-1 回归 — 摇杆持久流 / 拟人化分层 / 走位转向不断流。

背景（为什么需要这个文件）
──────────────────────────
旧实现每 tick 用 `self._last_action == action`（浮点元组相等）判"同向续流"，
方向差一个尾数就中断并重建摇杆宏；而重建的宏首事件前有 HUMANIZE_REACTION_MS
= 120ms 反应延迟 > DECISION_PERIOD_MS = 50ms 节拍 → 每个宏都在发出首个按下事件
之前就被下一 tick 打断。

实测后果：方向每 tick 变化 0.05° 时，2 秒内按下事件数 = 0，智能体完全不动。
而 README 的 `python -m core.orchestrator` 全 PASS —— 因为 Scene() 是静止的、
RulePolicy 输出逐帧完全相同，恰好走了续流快路径，demo 把 bug 盖住了。

所以本文件的核心断言不是"能动"，而是"**对手在动时也能动**"。
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from core.constants import (
    CONTACT_BUTTON, CONTACT_JOYSTICK, GAP_TOLERANCE_X, JOYSTICK_HZ,
)
from core.detectors import MockFastDetector
from core.executor import Executor
from core.injection import MockInjector, TouchEvent
from core.keymap import Keymap
from core.macro import Macro, MacroEngine, MonotonicClock, Tap
from core.orchestrator import Orchestrator
from core.perception import CVReader, Perception
from core.policy import Action, PolicyProvider
from core.screen_capture import SyntheticSource
from core.state import StateBuilder
from core.synthetic import Scene

PERIOD_MS = 1000.0 / JOYSTICK_HZ


# ═════════════════════════ 夹具与工具 ═════════════════════════
@pytest.fixture()
def km():
    return Keymap.load()


def stick_events(inj, contact: int = CONTACT_JOYSTICK):
    """该触点上的真实注入事件（不含 reset/release 记账）。"""
    return [e for e in inj.events
            if e.kind == "inject" and e.event.contact == contact]


class DriftPolicy(PolicyProvider):
    """对手一直在动 → 每 tick 追击方向都变 deg_step 度。

    这是旧实现下"完全不动"的最小复现条件：deg_step 只要不为 0，
    浮点相等判定就会失败。
    """
    name = "drift"

    def __init__(self, deg_step: float = 1.7, button: str | None = None):
        self.deg_step = deg_step
        self.button = button
        self.i = 0

    def decide(self, state: np.ndarray) -> Action:
        self.i += 1
        a = math.radians(self.i * self.deg_step)
        return Action(move=(math.cos(a), math.sin(a)), button=self.button)


def make_orch(policy, humanize: bool = False):
    km = Keymap.load()
    src = SyntheticSource(fps=60, scene=Scene())
    perc = Perception(MockFastDetector(), CVReader(km), slow_every=5)
    inj = MockInjector(clock=time.monotonic)
    ex = Executor(inj, km, humanize=humanize)
    return Orchestrator(src, perc, StateBuilder(), policy, ex), inj


# ═════════════════════════ JoystickStream 原语 ═════════════════════════
class TestStreamPrimitive:
    def test_press_once_and_keeps_contact_down(self, km):
        """启动后只按下一次, 且期间触点持续保持按下。"""
        from core.macro import JoystickStream
        inj = MockInjector(clock=time.monotonic)
        eng = MacroEngine(inj, clock=MonotonicClock(), humanize=False)
        cx, cy, r = km.circle("joystick")
        s = JoystickStream(eng, cx, cy, r)
        s.set_direction(1.0, 0.0)
        s.start()
        time.sleep(0.26)
        assert s.pressed and CONTACT_JOYSTICK in inj.contacts_pressed()
        s.stop(timeout=1.0)
        evs = stick_events(inj)
        assert sum(1 for e in evs if e.event.action == "d") == 1   # 只按一次
        assert sum(1 for e in evs if e.event.action == "m") >= 3   # 20Hz 续流
        assert inj.contacts_pressed() == set()                     # 停流零残留

    def test_set_direction_never_releases_contact(self, km):
        """反复改方向（含 180° 掉头）→ 触点全程不释放, 按下次数恒为 1。"""
        from core.macro import JoystickStream
        inj = MockInjector(clock=time.monotonic)
        eng = MacroEngine(inj, clock=MonotonicClock(), humanize=False)
        cx, cy, r = km.circle("joystick")
        s = JoystickStream(eng, cx, cy, r)
        s.set_direction(1.0, 0.0)
        s.start()
        try:
            for deg in range(0, 360, 15):
                a = math.radians(deg)
                s.set_direction(math.cos(a), math.sin(a))
                time.sleep(0.012)
            s.set_direction(-1.0, 0.0)          # 180° 掉头
            time.sleep(0.06)
            assert s.pressed
            assert CONTACT_JOYSTICK in inj.contacts_pressed()
        finally:
            s.stop(timeout=1.0)
        downs = [e for e in stick_events(inj) if e.event.action == "d"]
        assert len(downs) == 1, f"转向导致重新按下 {len(downs)} 次"
        # 掉头后推杆点应落到锚点左侧
        last_m = [e for e in stick_events(inj) if e.event.action == "m"][-1]
        assert last_m.event.x < cx

    def test_stop_releases_only_own_contact(self, km):
        """精确松手: 停摇杆流不得波及按钮那一路触点（取消后摇的前提）。"""
        from core.macro import JoystickStream
        inj = MockInjector(clock=time.monotonic)
        eng = MacroEngine(inj, clock=MonotonicClock(), humanize=False)
        cx, cy, r = km.circle("joystick")
        inj.inject([TouchEvent(CONTACT_BUTTON, 500, 500, "d")])   # 模拟按钮按住
        s = JoystickStream(eng, cx, cy, r)
        s.set_direction(0.0, -1.0)
        s.start()
        time.sleep(0.08)
        assert inj.contacts_pressed() == {CONTACT_JOYSTICK, CONTACT_BUTTON}
        s.stop(timeout=1.0)
        assert inj.contacts_pressed() == {CONTACT_BUTTON}          # 按钮未被牵连
        inj.reset()

    def test_first_press_keeps_humanize_reaction_delay(self, km):
        """拟人化: 首次按下属"反应型动作", 仍须保留 120ms 反应延迟。"""
        from core.constants import HUMANIZE_REACTION_MS
        from core.macro import JoystickStream
        inj = MockInjector(clock=time.monotonic)
        eng = MacroEngine(inj, clock=MonotonicClock(), humanize=True)
        cx, cy, r = km.circle("joystick")
        s = JoystickStream(eng, cx, cy, r)
        s.set_direction(1.0, 0.0)
        t0 = time.monotonic()
        s.start()
        time.sleep(0.05)
        assert stick_events(inj) == [], "120ms 反应延迟内不该有触点事件"
        time.sleep(0.16)
        s.stop(timeout=1.0)
        evs = stick_events(inj)
        assert evs, "反应延迟结束后应正常按下"
        assert (evs[0].ts - t0) * 1000 >= HUMANIZE_REACTION_MS * 0.9

    def test_zero_direction_centers_stick_but_holds_contact(self, km):
        """回中: 方向归零 → 停走但手指仍握杆（不产生 up/down 抖动）。"""
        from core.macro import JoystickStream
        inj = MockInjector(clock=time.monotonic)
        eng = MacroEngine(inj, clock=MonotonicClock(), humanize=False)
        cx, cy, r = km.circle("joystick")
        s = JoystickStream(eng, cx, cy, r)
        s.set_direction(1.0, 0.0)
        s.start()
        time.sleep(0.12)
        s.set_direction(0.0, 0.0)
        time.sleep(0.12)
        assert s.pressed and CONTACT_JOYSTICK in inj.contacts_pressed()
        s.stop(timeout=1.0)
        tail = [e for e in stick_events(inj) if e.event.action == "m"][-1]
        assert (tail.event.x, tail.event.y) == (round(cx), round(cy))


# ═════════════════════════ 拟人化分层 ═════════════════════════
class TestHumanizeLayering:
    """反应延迟只加在"反应型"宏上; 序列型宏必须零延迟（否则追不上帧级窗口）。"""

    def _engine(self, humanize=True, seed=7):
        from core.macro import VirtualClock
        clock = VirtualClock()
        inj = MockInjector(clock=clock.now)
        return MacroEngine(inj, clock=clock, rng_seed=seed, humanize=humanize), inj

    def test_sequence_macro_has_no_reaction_delay(self):
        eng, inj = self._engine(humanize=True)
        t0 = eng.clock.now()
        res = eng.play(Macro("连段内部", (Tap(100, 200, contact=CONTACT_BUTTON),),
                             humanize=False))
        assert res.start_ts == pytest.approx(t0, abs=1e-9)

    def test_reactive_macro_keeps_reaction_delay(self):
        from core.constants import HUMANIZE_REACTION_MS
        eng, inj = self._engine(humanize=True)
        t0 = eng.clock.now()
        res = eng.play(Macro("首次响应", (Tap(100, 200, contact=CONTACT_BUTTON),)))
        assert (res.start_ts - t0) >= HUMANIZE_REACTION_MS / 1000.0 * 0.99

    def test_sequence_macro_default_is_reactive(self):
        """默认值必须是 True —— 宪法「拟人化永远开启」, 显式声明才可豁免。"""
        assert Macro("m", (Tap(1, 2),)).humanize is True

    def test_miss_perturbation_survives_sequence_mode(self, monkeypatch):
        """1% 指令级失误与反应延迟是两回事: 序列型宏仍然带失误扰动。"""
        import core.macro as macro_mod
        monkeypatch.setattr(macro_mod, "HUMANIZE_MISS_RATE", 1.0)
        monkeypatch.setattr(macro_mod, "HUMANIZE_MISS_PX", 40.0)
        eng, inj = self._engine(humanize=True, seed=3)
        eng.play(Macro("连段内部", (Tap(600, 400, contact=CONTACT_BUTTON),),
                       humanize=False))
        first = stick_events(inj, CONTACT_BUTTON)[0].event
        assert (first.x, first.y) != (600, 400), "失误扰动不该被 humanize=False 关掉"


# ═════════════════════════ 主循环走位（★核心回归） ═════════════════════════
class TestOrchestratorSteering:
    @pytest.mark.parametrize("deg_step", [0.05, 1.7, 11.0])
    def test_drifting_direction_never_breaks_stick(self, deg_step):
        """对手在动 → 方向每 tick 都在变 → 摇杆必须持续按下且不断流。

        旧实现在 deg_step=0.05 时按下事件数 = 0（智能体完全不动）。
        """
        orch, inj = make_orch(DriftPolicy(deg_step=deg_step))
        rpt = orch.run(duration_s=1.2)
        evs = stick_events(inj)
        downs = [e for e in evs if e.event.action == "d"]
        moves = [e for e in evs if e.event.action == "m"]

        assert rpt.ticks >= 20
        assert len(downs) == 1, f"摇杆被按下 {len(downs)} 次（应只 1 次）"
        assert rpt.stick_starts == 1
        assert len(moves) >= 15, f"20Hz 续流事件只有 {len(moves)} 个"
        assert inj.contacts_pressed() == set()          # 收尾零残留

    def test_no_120ms_stall_between_stick_events(self):
        """相邻摇杆事件间隔不得出现 120ms 级停顿（旧 bug 的特征值）。"""
        orch, inj = make_orch(DriftPolicy(deg_step=3.0))
        orch.run(duration_s=1.2)
        ts = np.array([e.ts for e in stick_events(inj)])
        gaps = np.diff(ts) * 1000.0
        assert len(gaps) >= 15
        # 节拍 50ms; 容忍到 150ms 以吸收 CI 抖动, 但必须显著低于旧 bug 的
        # "永远等不到第一个事件" 形态, 也低于 2×拟人延迟。
        assert gaps.max() <= 150.0, f"最大间隔 {gaps.max():.1f}ms, 疑似断流"
        assert float(np.percentile(gaps, 95)) <= PERIOD_MS * GAP_TOLERANCE_X * 2

    def test_button_and_move_run_in_parallel(self):
        """边走边放技能: 摇杆(CONTACT_JOYSTICK)与按钮(CONTACT_BUTTON)分路互不顶掉。"""
        orch, inj = make_orch(DriftPolicy(deg_step=2.0, button="skill1"))
        rpt = orch.run(ticks=10)
        stick = stick_events(inj, CONTACT_JOYSTICK)
        btn = stick_events(inj, CONTACT_BUTTON)
        assert any(e.event.action == "d" for e in stick), "走位没发出去"
        assert any(e.event.action == "d" for e in btn), "按钮没发出去"
        assert rpt.action_counts.get("skill1") == 10
        assert rpt.stick_starts == 1, "按钮下发不该重建摇杆流"
        assert inj.contacts_pressed() == set()

    def test_button_does_not_tear_down_stick_contact(self):
        """按钮在飞期间, 摇杆触点必须一直保持按下。"""
        orch, inj = make_orch(DriftPolicy(deg_step=2.0, button="ult"))
        orch.run(ticks=6)
        evs = stick_events(inj, CONTACT_JOYSTICK)
        assert sum(1 for e in evs if e.event.action == "d") == 1
        assert not [e for e in evs if e.event.action == "u"], "摇杆被按钮动作松开了"

    def test_steady_direction_still_single_press(self):
        """恒定方向（旧 demo 唯一能过的场景）行为不回退: 仍是按下一次。"""
        from core.policy import Action as A

        class Fixed(PolicyProvider):
            name = "fixed"

            def decide(self, s):
                return A(move=(1.0, 0.0))

        orch, inj = make_orch(Fixed())
        rpt = orch.run(ticks=12)
        evs = stick_events(inj)
        assert sum(1 for e in evs if e.event.action == "d") == 1
        assert rpt.stick_starts == 1
        # 占空比: 旧实现在恒定方向下也只有 62.5%(每 320ms 断流 120ms),
        # 修复后应是连续 20Hz 流。
        assert sum(1 for e in evs if e.event.action == "m") >= 10

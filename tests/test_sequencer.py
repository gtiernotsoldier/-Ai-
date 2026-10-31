# -*- coding: utf-8 -*-
"""M4.6 单测 + 端到端 — Sequencer 帧级执行层。

核心回归(对比旧 Orchestrator 用法):
  - 连段一次编译一次下发, 20Hz 刷屏不会逐 tick 重建/拆散连段
  - 取消窗口内替换 = 取消后摇; 窗口未开时拒绝
  - ⑦ 事件分支: bait 在飞 + sub_flash 边沿 → 打断剩余步骤, 帧级切 punish
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from core.action_state import ActionPhase
from core.detectors import MockFastDetector
from core.executor import Executor
from core.frame_data import FrameData
from core.injection import MockInjector
from core.keymap import Keymap
from core.orchestrator import Orchestrator
from core.perception import CVReader, Perception
from core.policy import Action, PolicyProvider
from core.screen_capture import SyntheticSource
from core.sequencer import (MIN_STEP_GAP_S, Sequence, Sequencer, Step,
                            combo_sequence)
from core.state import StateBuilder
from core.synthetic import Scene

# fps=100 → 10ms/帧; atk: startup2 active2 recovery6 total10 cancel_from4
MINI_RAW = {
    "fps": 100,
    "actions": {
        "atk":    {"startup": 2, "active": 2, "recovery": 6, "cancel_from": 4,
                   "cancel_into": ["atk", "skill1"]},
        "skill1": {"startup": 3, "active": 2, "recovery": 2, "cancel_from": 6,
                   "cancel_into": []},
        "ult":    {"startup": 5, "active": 5, "recovery": 5, "cancel_from": 15,
                   "cancel_into": []},
    },
    "combos": {
        "basic_press": {"steps": ["atk", "atk", "atk"]},
        "bait_atk":    {"steps": ["atk", "atk"]},
        "punish_sub":  {"steps": ["skill1", "atk"]},
    },
}


def make_seqr(mini=True, humanize=False):
    fd = FrameData(MINI_RAW) if mini else FrameData.load()
    inj = MockInjector(clock=time.monotonic)
    ex = Executor(inj, Keymap.load(), humanize=humanize)
    seqr = Sequencer(ex, fd)
    return seqr, inj, ex


def downs(inj, button=None):
    evs = [e.event for e in inj.events if e.kind == "inject"
           and e.event.action == "d"]
    if button is None:
        return evs
    cx, cy, _ = Keymap.load().circle(f"btn_{button}")
    return [e for e in evs if (e.x, e.y) == (cx, cy)]


class TestCompile:
    def test_macro_is_sequence_type(self):
        seqr, _, _ = make_seqr()
        seq = combo_sequence(seqr.fd, "basic_press")
        m = seqr._compile(seq)
        assert m.humanize is False            # 序列型: 连段内部零拟人延迟
        assert m.name == "seq_basic_press"

    def test_step_gap_respects_frame_data(self):
        seqr, _, _ = make_seqr()
        seq = combo_sequence(seqr.fd, "basic_press")
        # atk→atk: 帧表 gap=4帧=40ms, 但同触点保护 hold(40ms)+20ms=60ms 取大
        expected = max(0.04, 0.04 + MIN_STEP_GAP_S)
        assert abs(seqr._step_gap_s(seq.steps[0], seq.steps[1]) - expected) < 1e-9

    def test_plan_presses_match_schedule(self):
        seqr, _, _ = make_seqr()
        seq = combo_sequence(seqr.fd, "bait_atk")     # atk→atk
        plan = seqr._plan_presses(seq, seqr._clock.now())
        assert len(plan) == 2
        assert abs((plan[1][0] - plan[0][0])
                   - seqr._step_gap_s(seq.steps[0], seq.steps[1])) < 1e-6


class TestSubmit:
    def test_submit_when_idle(self):
        seqr, inj, _ = make_seqr()
        try:
            assert seqr.submit(combo_sequence(seqr.fd, "bait_atk"))
            assert seqr.stats["submitted"] == 1
        finally:
            seqr.shutdown()

    def test_busy_rejects_without_window(self):
        seqr, _, _ = make_seqr()
        try:
            seqr.submit(combo_sequence(seqr.fd, "bait_atk"))
            # 第一步刚按下(frame≈0 < cancel_from=4) → 窗口未开 → 拒绝
            assert not seqr.submit(combo_sequence(seqr.fd, "basic_press"))
            assert seqr.stats["rejected_busy"] == 1
            assert seqr.stats["cancel_swap"] == 0
        finally:
            seqr.shutdown()

    def test_cancel_window_swap(self):
        seqr, inj, _ = make_seqr()
        try:
            seqr.submit(combo_sequence(seqr.fd, "bait_atk"))   # atk
            # 轮询到取消窗口开(mini 表: cancel_from 4帧=40ms), 不依赖 sleep 精度
            deadline = time.monotonic() + 0.2
            while not seqr.tracker.cancel_window_open(seqr._clock.now(), "skill1"):
                assert time.monotonic() < deadline, "取消窗口未按时打开"
                assert not seqr._handle.done, "宏过早结束"
                time.sleep(0.005)
            assert seqr.submit(combo_sequence(seqr.fd, "punish_sub"))
            assert seqr.stats["cancel_swap"] == 1
            time.sleep(0.15)
            assert len(downs(inj, "skill1")) >= 1   # 新序列真的接管了
        finally:
            seqr.shutdown()

    def test_force_bypasses_window(self):
        seqr, _, _ = make_seqr()
        try:
            seqr.submit(combo_sequence(seqr.fd, "bait_atk"))
            # ult 不在 atk.cancel_into, 窗口语义下不可换; force 无条件打断
            assert seqr.submit(combo_sequence(seqr.fd, "punish_sub"), force=True)
            assert seqr.stats["cancel_swap"] == 0
        finally:
            seqr.shutdown()


class TestRealTiming:
    """MonotonicClock 真跑: 三连击的按下时刻按帧表排程。"""

    def test_three_hit_combo_down_times(self):
        seqr, inj, _ = make_seqr()
        try:
            seqr.submit(combo_sequence(seqr.fd, "basic_press"))
            time.sleep(0.30)
            ds = downs(inj, "atk")
            assert len(ds) == 3                     # 一次下发, 三步都触发
            inj_ts = [e.ts for e in inj.events
                      if e.kind == "inject" and e.event.action == "d"]
            assert len(inj_ts) == 3
            gap = seqr._step_gap_s(Step("atk"), Step("atk"))
            for a, b in zip(inj_ts, inj_ts[1:]):
                assert gap - 0.03 <= b - a <= gap + 0.10   # ≈60ms ±调度抖动
        finally:
            seqr.shutdown()


class TestEventBranch:
    """⑦ 卡替身: bait 在飞 + enemy_sub 边沿 → 帧级切 punish。"""

    def test_branch_requires_running_sequence(self):
        seqr, _, _ = make_seqr()
        assert seqr.on_events(["enemy_sub"]) is False

    def test_undeclared_event_ignored(self):
        seqr, _, _ = make_seqr()
        try:
            bait = combo_sequence(seqr.fd, "bait_atk",
                                  branches=(("enemy_sub", "punish_sub"),))
            seqr.submit(bait)
            assert seqr.on_events(["enemy_ult"]) is False
            assert seqr.stats["branch_taken"] == 0
        finally:
            seqr.shutdown()

    def test_branch_interrupts_remaining_steps(self):
        seqr, inj, _ = make_seqr()
        try:
            bait = combo_sequence(seqr.fd, "bait_atk",
                                  branches=(("enemy_sub", "punish_sub"),))
            seqr.submit(bait)
            time.sleep(0.02)                # 第一步在飞, 第二步未按
            assert seqr.on_events(["enemy_sub"]) is True
            assert seqr.stats["branch_taken"] == 1
            time.sleep(0.15)
            # punish 序列接管: skill1(首步) 按下; bait 第二步被吃掉;
            # punish 第二步 atk 在 skill1 打满 total(7帧=70ms)后按下
            assert seqr.current_name == "punish_sub"
            assert len(downs(inj, "skill1")) == 1
            assert len(downs(inj, "atk")) == 2       # bait 第一步 + punish 第二步
        finally:
            seqr.shutdown()


# ═════════════════ Orchestrator 集成 ═════════════════
class FixedPolicy(PolicyProvider):
    name = "fixed"

    def __init__(self, action: Action):
        self.action = action

    def decide(self, state: np.ndarray) -> Action:
        return self.action


class BaitPolicy(PolicyProvider):
    """⑦ 意图式策略替身: 周期性提交 bait 序列(带 sub→punish 分支)。"""
    name = "bait"

    def __init__(self, fd: FrameData, every: int = 6):
        self.fd = fd
        self.every = every
        self.n = 0
        self.seqr: Sequencer | None = None

    def bind_sequencer(self, s: Sequencer) -> None:
        self.seqr = s

    def decide(self, state: np.ndarray) -> Action:
        self.n += 1
        if self.seqr is not None and self.n % self.every == 1:
            self.seqr.submit(combo_sequence(
                self.fd, "bait_atk", branches=(("enemy_sub", "punish_sub"),)))
        return Action(move=(0.3, 0.0), strength=0.5)    # 慢速逼近


def make_orch(policy, scene):
    src = SyntheticSource(fps=60, scene=scene)
    km = Keymap.load()
    perc = Perception(MockFastDetector(), CVReader(km), slow_every=5)
    inj = MockInjector(clock=time.monotonic)
    ex = Executor(inj, km, humanize=True)
    seqr = Sequencer(ex, FrameData.load())
    orch = Orchestrator(src, perc, StateBuilder(), policy, ex, sequencer=seqr)
    return orch, inj, seqr


class TestOrchIntegration:
    def test_tick_spam_cannot_shred_combo(self):
        """20Hz 每 tick 刷同一按钮: 不再逐 tick 重建宏(旧 bug 形态=down≈ticks)。"""
        orch, inj, seqr = make_orch(FixedPolicy(Action(button="atk")), Scene())
        rpt = orch.run(ticks=8)                     # 400ms
        n = len(downs(inj, "atk"))
        assert 2 <= n <= 5                          # 生产表 atk 宏≈267ms → ~2次
        assert rpt.seq_stats["rejected_busy"] + rpt.seq_stats["cancel_swap"] >= 1
        assert rpt.seq_stats["submitted"] == n      # 每次真实 down=一次提交

    def test_resid_clean_after_run(self):
        orch, inj, seqr = make_orch(BaitPolicy(FrameData.load()), Scene())
        orch.run(duration_s=1.0)
        assert inj.contacts_pressed() == set()
        seqr.shutdown()

    def test_bait_to_punish_end_to_end(self):
        """bait 在飞 → 合成帧出现替身闪 → 感知边沿 → Sequencer 帧级切 punish。"""
        scene = Scene()
        orch, inj, seqr = make_orch(BaitPolicy(FrameData.load(), every=6), scene)
        # t=0.35s 敌方替身闪(合成场景真值 → MockFastDetector → effect_sub)
        threading.Timer(0.35, scene.effects.add, args=("sub_flash_enemy",)).start()
        threading.Timer(0.90, scene.effects.discard,
                        args=("sub_flash_enemy",)).start()
        rpt = orch.run(duration_s=1.2)
        assert rpt.seq_stats["branch_taken"] == 1
        assert len(downs(inj, "skill2")) == 1       # punish 首步真的发出
        assert len(downs(inj, "atk")) >= 3          # bait 起手 + punish 连段
        assert inj.contacts_pressed() == set()      # 中途打断+收尾 → 零残留

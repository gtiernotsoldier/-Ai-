# -*- coding: utf-8 -*-
"""M4.5 单测 — 决策接口 (policy): 动作空间 / 规则策略 / Onnx 优雅降级。"""
from __future__ import annotations

import numpy as np
import pytest

from core.exceptions import DependencyMissing
from core.policy import (
    ACTION_BUTTONS, Action, OnnxPolicy, RulePolicy, SlowPolicy, StateReader,
    action_from_index,
)
from core.state import DEFAULT_LAYOUT, StateLayout


def make_state() -> np.ndarray:
    return np.zeros(64, dtype=np.float32)


def put(state: np.ndarray, slice_name: str, field: str, value: float) -> None:
    s = next(s for s in DEFAULT_LAYOUT.SLICES if s.name == slice_name)
    state[s.start + s.fields.index(field)] = value


class TestAction:
    def test_default_idle(self):
        a = Action()
        assert a.move == (0.0, 0.0) and a.button is None

    def test_invalid_button(self):
        with pytest.raises(ValueError):
            Action(button="nope")

    def test_invalid_move_norm(self):
        with pytest.raises(ValueError):
            Action(move=(2.0, 0.0))

    def test_unit_move_ok(self):
        Action(move=(0.6, -0.8))
        Action(move=(1.0, 0.0))


class TestActionTable:
    def test_move_indices(self):
        assert action_from_index(0).move == (0.0, 0.0)
        assert action_from_index(3).move == (1.0, 0.0)

    def test_button_indices(self):
        for i, b in enumerate(ACTION_BUTTONS):
            assert action_from_index(9 + i).button == b

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            action_from_index(99)


class TestStateReader:
    def test_field_lookup_matches_layout(self):
        r = StateReader()
        st = make_state()
        put(st, "resources", "hp_self", 0.77)
        assert r(st, "resources", "hp_self") == pytest.approx(0.77)

    def test_layout_coverage_unchanged(self):
        StateLayout()  # 断言内部自检: 64 维无重叠无遗漏


class TestRulePolicy:
    def test_sub_when_low_hp(self):
        st = make_state()
        put(st, "resources", "hp_self", 0.2)
        put(st, "resources", "sub_ready_self", 1.0)
        assert RulePolicy().decide(st).button == "sub"

    def test_ult_when_energy_full(self):
        st = make_state()
        put(st, "resources", "hp_self", 0.9)
        put(st, "resources", "energy_self", 0.8)
        put(st, "resources", "ult_cd_self", 0.0)
        put(st, "enemy_kin", "visible", 1.0)
        put(st, "enemy_kin", "dx", 0.6)
        put(st, "enemy_kin", "dy", 0.5)
        put(st, "self_kin", "wall_dx", 0.4)
        put(st, "self_kin", "wall_dy", 0.5)
        put(st, "relative", "dist", 0.4)          # 600px: 不近身
        assert RulePolicy().decide(st).button == "ult"

    def test_skill1_when_close(self):
        st = make_state()
        put(st, "resources", "hp_self", 0.9)
        put(st, "resources", "energy_self", 0.2)
        put(st, "enemy_kin", "visible", 1.0)
        put(st, "enemy_kin", "dx", 0.6)
        put(st, "self_kin", "wall_dx", 0.5)
        put(st, "relative", "dist", 0.15)         # 225px < 300
        assert RulePolicy().decide(st).button == "skill1"

    def test_move_toward_enemy(self):
        st = make_state()
        put(st, "resources", "hp_self", 0.9)
        put(st, "skill_cds", "skill1", 0.5)      # 绕过近身技能分支
        put(st, "enemy_kin", "visible", 1.0)
        put(st, "enemy_kin", "dx", 0.7)
        put(st, "enemy_kin", "dy", 0.3)
        put(st, "self_kin", "wall_dx", 0.3)
        put(st, "self_kin", "wall_dy", 0.5)
        put(st, "relative", "dist", 0.5)
        a = RulePolicy().decide(st)
        assert a.button is None
        assert a.move[0] > 0.5 and a.move[1] < 0.0  # 720 高度跨度小, y 分量较小

    def test_retreat_when_engaged(self):
        st = make_state()
        put(st, "resources", "hp_self", 0.9)
        put(st, "skill_cds", "skill1", 0.5)      # 绕过近身技能分支
        put(st, "enemy_kin", "visible", 1.0)
        put(st, "enemy_kin", "dx", 0.7)
        put(st, "enemy_kin", "dy", 0.5)          # v3: 0.5=无纵向偏移
        put(st, "self_kin", "wall_dx", 0.5)
        put(st, "relative", "dist", 0.05)         # 75px < 120 近身圈
        assert RulePolicy().decide(st).move[0] < -0.5   # 反向拉开

    def test_idle_when_enemy_gone(self):
        st = make_state()
        put(st, "resources", "hp_self", 0.9)
        a = RulePolicy().decide(st)
        assert a.move == (0.0, 0.0) and a.button is None


class TestGuards:
    def test_onnx_graceful_in_sandbox(self):
        """沙箱无 onnxruntime → DependencyMissing（不崩进程）。"""
        pytest.importorskip("shutil")
        import builtins
        real_import = builtins.__import__

        def no_ort(name, *a, **k):
            if name == "onnxruntime":
                raise ImportError("blocked")
            return real_import(name, *a, **k)

        builtins.__import__, orig = no_ort, real_import
        try:
            with pytest.raises(DependencyMissing):
                OnnxPolicy()
        finally:
            builtins.__import__ = orig

    def test_slow_policy_for_watchdog(self):
        st = make_state()
        sp = SlowPolicy(RulePolicy(), delay_s=0.02)
        import time as _t
        t0 = _t.perf_counter()
        sp.decide(st)
        assert _t.perf_counter() - t0 >= 0.02

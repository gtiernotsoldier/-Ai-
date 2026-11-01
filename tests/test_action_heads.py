# -*- coding: utf-8 -*-
"""动作空间四头（Shūkai）: 组合表达力 / 施法朝向 / 合法性屏蔽 / 时长档位。"""
from __future__ import annotations

import numpy as np
import pytest

from core.policy import (
    ACTION_BUTTONS, ActionHeads, ActionMask, DURATION_BUCKETS_MS, OnnxPolicy,
    quantize_duration,
)
from core.state import DEFAULT_LAYOUT


def put(state: np.ndarray, slice_name: str, field: str, value: float) -> None:
    s = next(s for s in DEFAULT_LAYOUT.SLICES if s.name == slice_name)
    state[s.start + s.fields.index(field)] = value


class TestActionHeads:
    def test_move_only(self):
        assert ActionHeads(move_ud=0, move_lr=1).move() == (0.0, -1.0)
        assert ActionHeads(move_ud=2, move_lr=2).move() == pytest.approx(
            (0.7071, 0.7071), abs=1e-3)

    def test_idle(self):
        assert ActionHeads().move() == (0.0, 0.0)
        assert ActionHeads().button is None

    def test_move_and_skill_simultaneously(self):
        """旧 17 选 1 动作表的死结: 移动与按钮互斥, 这里必须两者并存。"""
        a = ActionHeads(move_ud=0, move_lr=2, skill=6).to_action()   # 右上 + atk
        assert a.button == "atk"
        assert a.move[0] > 0 and a.move[1] < 0

    def test_cast_direction_overrides_move(self):
        """施法朝向 = 推杆方向: 有 direction 时它接管 move。"""
        h = ActionHeads(move_ud=0, move_lr=1, skill=1, direction=5)   # 朝下放技能1
        assert h.move() == (0.0, 1.0)
        assert h.button == "skill1"

    def test_direction_ignored_without_skill(self):
        h = ActionHeads(move_ud=1, move_lr=2, skill=0, direction=5)
        assert h.move() == (1.0, 0.0)

    @pytest.mark.parametrize("kw", [
        {"move_ud": 3}, {"move_lr": -1}, {"skill": len(ACTION_BUTTONS) + 1},
        {"direction": 9},
    ])
    def test_out_of_range_rejected(self, kw):
        with pytest.raises(ValueError):
            ActionHeads(**kw)


class TestDurationBuckets:
    def test_quantize_nearest(self):
        assert quantize_duration(45) == 40.0
        assert quantize_duration(190) == 200.0
        assert quantize_duration(999) == 500.0

    def test_to_action_quantizes(self):
        a = ActionHeads(skill=1, duration_ms=310).to_action()
        assert a.duration_ms == 320.0
        assert a.duration_ms in DURATION_BUCKETS_MS


class TestActionMask:
    def _ready_state(self) -> np.ndarray:
        st = np.zeros(64, dtype=np.float32)
        put(st, "resources", "hp_self", 0.9)
        put(st, "resources", "energy_self", 0.8)
        put(st, "resources", "sub_ready_self", 1.0)
        return st

    def test_ready_state_allows_everything(self):
        m = ActionMask.from_state(self._ready_state())
        assert all(m.skill), "资源齐备时不该屏蔽任何技能"

    def test_ult_needs_energy(self):
        st = self._ready_state()
        put(st, "resources", "energy_self", 0.1)
        m = ActionMask.from_state(st)
        i = 1 + ACTION_BUTTONS.index("ult")
        assert m.skill[i] is False

    def test_skill_on_cd_masked(self):
        st = self._ready_state()
        put(st, "skill_cds", "skill2", 0.5)
        m = ActionMask.from_state(st)
        assert m.skill[1 + ACTION_BUTTONS.index("skill2")] is False

    def test_sub_needs_ready(self):
        st = self._ready_state()
        put(st, "resources", "sub_ready_self", 0.0)
        m = ActionMask.from_state(st)
        assert m.skill[1 + ACTION_BUTTONS.index("sub")] is False

    def test_cc_masks_everything_but_idle(self):
        st = np.zeros(64, dtype=np.float32)
        put(st, "effects", "cc", 1.0)
        m = ActionMask.from_state(st)
        assert m.move_ud == (False, True, False)
        assert m.skill == (True,) + (False,) * len(ACTION_BUTTONS)

    def test_skill_logits_sets_neg_inf(self):
        st = np.zeros(64, dtype=np.float32)
        put(st, "skill_cds", "skill1", 0.5)        # 只有 skill1 在 CD
        m = ActionMask.from_state(st)
        out = m.skill_logits(np.zeros(len(ACTION_BUTTONS) + 1, dtype=np.float32))
        assert out[0] == 0.0                       # idle 永远合法
        assert out[1] < -1e8                       # skill1 被屏蔽


class TestActionChunk:
    def test_sequence_from_chunk(self):
        from core.policy import ActionChunk
        ch = ActionChunk((ActionHeads(skill=6, hold_ms=40),
                          ActionHeads(skill=6, hold_ms=80),
                          ActionHeads(skill=1, hold_ms=40)))
        seq = ch.to_sequence("combo")
        assert [s.button for s in seq.steps] == ["atk", "atk", "skill1"]
        assert [s.hold_ms for s in seq.steps] == [40, 80, 40]

    def test_move_only_chunk_cannot_be_sequence(self):
        from core.policy import ActionChunk
        ch = ActionChunk((ActionHeads(move_ud=0), ActionHeads(move_lr=0)))
        with pytest.raises(ValueError):
            ch.to_sequence()

    def test_bounds(self):
        from core.policy import MAX_CHUNK, ActionChunk
        with pytest.raises(ValueError):
            ActionChunk(())
        with pytest.raises(ValueError):
            ActionChunk(tuple(ActionHeads(skill=1) for _ in range(MAX_CHUNK + 1)))

    def test_duration_sums(self):
        from core.policy import ActionChunk
        ch = ActionChunk((ActionHeads(duration_ms=200),
                          ActionHeads(duration_ms=320)))
        assert ch.duration_ms == 520.0

    def test_sequencer_compiles_chunk_once(self):
        """分块 = 一次编译一次下发: 3 步连段只应产生 3 个 tap 宏片段。"""
        from core.executor import Executor
        from core.frame_data import FrameData
        from core.injection import MockInjector
        from core.keymap import Keymap
        from core.macro import VirtualClock
        from core.policy import ActionChunk
        from core.sequencer import Sequencer

        clock = VirtualClock()
        inj = MockInjector(clock=clock.now)
        ex = Executor(inj, Keymap.load(), clock=clock, humanize=False)
        seqr = Sequencer(ex, FrameData.load(), clock=clock)
        ch = ActionChunk((ActionHeads(skill=6), ActionHeads(skill=6),
                          ActionHeads(skill=1)))
        assert seqr.submit(ch.to_sequence("combo")) is True
        assert seqr.stats["submitted"] == 1
        assert seqr.current_name == "combo"


class TestOnnxDecode:
    """不经 onnxruntime, 直接验证三种导出形态的解码分支。"""

    def _policy(self) -> OnnxPolicy:
        p = object.__new__(OnnxPolicy)           # 跳过 __init__ 的 ort 依赖
        p.duration_ms = 200.0
        return p

    def _state(self) -> np.ndarray:
        st = np.zeros(64, dtype=np.float32)
        put(st, "resources", "hp_self", 0.9)
        put(st, "resources", "energy_self", 0.8)
        put(st, "resources", "sub_ready_self", 1.0)
        return st

    def test_four_heads(self):
        p = self._policy()
        outs = [np.array([[0, 1, 2]]),            # move_ud → 下
                np.array([[0, 1, 2]]),            # move_lr → 右
                np.array([[0] * 9]),              # skill → 0
                np.array([[0] * 9])]              # direction → 0
        a = p._decode(outs, ActionMask.from_state(self._state()))
        assert a.button is None
        assert a.move[0] > 0 and a.move[1] > 0

    def test_concatenated_single_output(self):
        p = self._policy()
        flat = np.zeros((1, OnnxPolicy.TOTAL_HEAD_LOGITS), dtype=np.float32)
        flat[0, 0] = 5.0                          # move_ud = 上
        flat[0, 5] = 5.0                          # move_lr = 右
        a = p._decode([flat], ActionMask.from_state(self._state()))
        assert a.move[0] > 0 and a.move[1] < 0

    def test_legacy_single_head(self):
        p = self._policy()
        flat = np.zeros((1, 17), dtype=np.float32)
        flat[0, 3] = 5.0                          # 旧表: 3 = 右
        a = p._decode([flat], ActionMask.from_state(self._state()))
        assert a.move == (1.0, 0.0) and a.button is None

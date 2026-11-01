# -*- coding: utf-8 -*-
"""奖励函数（Shūkai）: 血量差 + 终局×7 + 三种人格权重。"""
from __future__ import annotations

import numpy as np
import pytest

from core.reward import (AGGRESSIVE, BALANCED, CAUTIOUS, PERSONALITIES,
                         TERMINAL_MULT, HpReward, Personality, hp_delta,
                         step_reward)
from core.state import DEFAULT_LAYOUT


class TestHpDelta:
    def test_enemy_damage_is_positive(self):
        assert hp_delta(1.0, 1.0, 1.0, 0.9) == pytest.approx(0.1)

    def test_self_damage_is_negative(self):
        assert hp_delta(1.0, 0.9, 1.0, 1.0) == pytest.approx(-0.1)

    def test_trade_even_is_zero(self):
        assert hp_delta(1.0, 0.9, 1.0, 0.9) == pytest.approx(0.0)

    def test_terminal_multiplier(self):
        base = step_reward(1.0, 0.9, 1.0, 0.9)
        term = step_reward(1.0, 0.9, 1.0, 0.9, terminal=True)
        assert term == pytest.approx(base * TERMINAL_MULT)


class TestPersonality:
    def test_cautious_fears_self_damage_more(self):
        b = step_reward(1.0, 0.9, 1.0, 1.0, personality=BALANCED)
        c = step_reward(1.0, 0.9, 1.0, 1.0, personality=CAUTIOUS)
        assert c < b < 0

    def test_aggressive_values_enemy_damage_more(self):
        b = step_reward(1.0, 1.0, 1.0, 0.9, personality=BALANCED)
        a = step_reward(1.0, 1.0, 1.0, 0.9, personality=AGGRESSIVE)
        assert a > b > 0

    def test_registry(self):
        assert set(PERSONALITIES) == {"balanced", "cautious", "aggressive"}

    def test_invalid_weight(self):
        with pytest.raises(ValueError):
            Personality("bad", self_damage_weight=0.0)


class TestHpRewardTracker:
    def _state(self, hp_self: float, hp_enemy: float) -> np.ndarray:
        v = np.zeros(64, dtype=np.float32)
        for s in DEFAULT_LAYOUT.SLICES:
            if s.name == "resources":
                v[s.start + s.fields.index("hp_self")] = hp_self
                v[s.start + s.fields.index("hp_enemy")] = hp_enemy
        return v

    def test_first_update_is_zero(self):
        t = HpReward()
        assert t.update(self._state(1.0, 1.0)) == 0.0

    def test_accumulates_deltas(self):
        t = HpReward()
        t.update(self._state(1.0, 1.0))
        assert t.update(self._state(0.9, 0.8)) == pytest.approx(0.1)  # 双掉血 -0.1+0.2

    def test_reset_clears_baseline(self):
        t = HpReward()
        t.update(self._state(1.0, 1.0))
        t.update(self._state(0.9, 1.0))
        t.reset()
        assert t.update(self._state(0.8, 1.0)) == 0.0

    def test_terminal_flag(self):
        t = HpReward()
        t.update(self._state(1.0, 1.0))
        r = t.update(self._state(1.0, 0.9), terminal=True)
        assert r == pytest.approx(0.1 * TERMINAL_MULT)

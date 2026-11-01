# -*- coding: utf-8 -*-
"""难度档 (HumanProfile): 延迟 / 执行概率 / 热切换 —— Shūkai 的难度设计。"""
from __future__ import annotations

import pytest

from core.human_profile import (ADVANCED, BEGINNER, DEFAULT_PROFILE,
                                INTERMEDIATE, PROFILES, HumanProfile,
                                get_profile)
from core.injection import MockInjector
from core.macro import Macro, MacroEngine, Tap, VirtualClock


def make_engine(profile: HumanProfile | None, humanize: bool = True):
    clock = VirtualClock()
    inj = MockInjector(clock=clock.now)
    eng = MacroEngine(inj, clock=clock, rng_seed=7, humanize=humanize,
                      profile=profile)
    return eng, inj, clock


def tap_macro() -> Macro:
    return Macro("tap", (Tap(100, 200, hold_ms=40),))


class TestProfiles:
    def test_three_tiers_ordered(self):
        assert BEGINNER.delay_ms > INTERMEDIATE.delay_ms > ADVANCED.delay_ms
        assert BEGINNER.exec_prob < INTERMEDIATE.exec_prob < ADVANCED.exec_prob

    def test_advanced_matches_legacy(self):
        from core.constants import HUMANIZE_MISS_RATE, HUMANIZE_REACTION_MS
        assert ADVANCED.delay_ms == HUMANIZE_REACTION_MS
        assert ADVANCED.miss_rate == HUMANIZE_MISS_RATE
        assert DEFAULT_PROFILE is ADVANCED

    def test_lookup_by_name(self):
        assert get_profile("beginner") is BEGINNER
        with pytest.raises(ValueError):
            get_profile("godlike")

    @pytest.mark.parametrize("kw", [
        {"name": "x", "delay_ms": -1},
        {"name": "x", "delay_ms": 100, "exec_prob": 1.5},
        {"name": "x", "delay_ms": 100, "miss_rate": -0.1},
    ])
    def test_invalid_rejected(self, kw):
        with pytest.raises(ValueError):
            HumanProfile(**kw)


class TestEngineIntegration:
    def test_none_profile_keeps_120ms(self):
        eng, inj, clock = make_engine(None)
        t0 = clock.now()
        res = eng.play(tap_macro())
        assert res.start_ts - t0 >= 0.119

    def test_beginner_slower(self):
        eng, _, clock = make_engine(BEGINNER)
        t0 = clock.now()
        res = eng.play(tap_macro())
        assert res.start_ts - t0 >= 0.219

    def test_zero_exec_prob_drops_instruction(self):
        """exec_prob=0 → 这条指令根本没发出去（人走神）。"""
        eng, inj, _ = make_engine(HumanProfile("ghost", 120.0, exec_prob=0.0))
        res = eng.play(tap_macro())
        assert res.n_events == 0
        assert not inj.events

    def test_full_exec_prob_keeps_events(self):
        eng, inj, _ = make_engine(HumanProfile("robot", 120.0, exec_prob=1.0))
        res = eng.play(tap_macro())
        assert res.n_events == 2 and len(inj.events) == 2

    def test_hot_swap_profile(self):
        eng, _, clock = make_engine(None)
        eng.set_profile(BEGINNER)
        t0 = clock.now()
        res = eng.play(tap_macro())
        assert res.start_ts - t0 >= 0.219

    def test_profiles_registry_exposes_all(self):
        assert set(PROFILES) == {"beginner", "intermediate", "advanced"}

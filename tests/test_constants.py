# -*- coding: utf-8 -*-
"""宪法红线与延迟预算表的静态一致性检查。"""
import core.constants as C


def test_latency_budget_sum():
    # 决策前向在表中取 15ms 上限(看门狗口径) → 最坏合计 74ms; 文档口径 65ms
    assert sum(b for _, b, _ in C.LATENCY_BUDGET) == C.BUDGET_WORST_CASE_MS == 74.0
    assert C.BUDGET_TOTAL_MS == 65.0


def test_red_lines():
    assert C.REDLINE_LATENCY_MS == 100
    assert C.TARGET_LATENCY_MS == 40
    assert C.HUMANIZE_REACTION_MS == 120
    assert 0 < C.HUMANIZE_MISS_RATE <= 0.05


def test_frequency_spec():
    assert C.FAST_HZ == 60 and C.SLOW_HZ == 12 and C.DECISION_HZ == 20
    assert C.SLOW_EVERY == C.FAST_HZ // C.SLOW_HZ == 5


def test_state_dim():
    assert C.STATE_DIM == 64

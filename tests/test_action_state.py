# -*- coding: utf-8 -*-
"""M4.6 单测 — 动作时序状态机 (action_state): 阶段推进/取消窗口/取消衔接/pending。

mini 帧表(fps=100 → 10ms/帧, 时间数字好算):
  atk    startup=2 active=2 recovery=6 total=10 cancel_from=4  cancel_into=(atk, skill1)
  skill1 startup=3 active=2 recovery=2 total=7  cancel_from=6  cancel_into=()
  ult    startup=5 active=5 recovery=5 total=15 cancel_from=15 cancel_into=()
"""
from __future__ import annotations

import pytest

from core.action_state import ActionPhase, ActionTracker
from core.frame_data import FrameData

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
    "combos": {},
}
# 注意 skill1.total=7 帧=70ms < atk→skill1 取消 gap(4帧=40ms? 不, gap_to=
# cancel_from=4帧) —— 为造 pending 场景改用 atk→ult(打满 gap=total_atk=100ms,
# skill1 若作第一步 total=70ms → 有 30ms pending 窗口)


@pytest.fixture()
def tr():
    return ActionTracker(FrameData(MINI_RAW))


def press(ts: float, button: str):
    return (ts, button)


class TestSingleMove:
    def test_empty_plan_idle(self, tr):
        q = tr.query(now=100.0)
        assert q.phase is ActionPhase.IDLE and q.button is None
        assert not tr.busy(100.0)

    def test_startup_active_recovery(self, tr):
        tr.begin((press(100.0, "atk"),))
        # frame 0-1: startup (startup=2)
        assert tr.query(100.0).phase is ActionPhase.STARTUP
        assert tr.query(100.019).phase is ActionPhase.STARTUP
        # frame 2-3: active (startup+active=4)
        assert tr.query(100.020).phase is ActionPhase.ACTIVE
        assert tr.query(100.039).phase is ActionPhase.ACTIVE
        # frame 4-9: recovery (total=10)
        assert tr.query(100.040).phase is ActionPhase.RECOVERY
        assert tr.query(100.099).phase is ActionPhase.RECOVERY
        # frame ≥10: 打完回 IDLE
        assert tr.query(100.100).phase is ActionPhase.IDLE
        assert not tr.busy(100.100)

    def test_frame_numbering(self, tr):
        tr.begin((press(100.0, "atk"),))
        assert tr.query(100.0).frame == 0
        assert tr.query(100.035).frame == 3

    def test_busy_during_move(self, tr):
        tr.begin((press(100.0, "atk"),))
        assert tr.busy(100.0)
        assert tr.busy(100.05)


class TestCancelWindow:
    def test_window_opens_at_cancel_from(self, tr):
        tr.begin((press(100.0, "atk"),))            # cancel_from=4
        assert not tr.query(100.039).cancel_open    # frame 3 < 4
        assert tr.query(100.040).cancel_open        # frame 4 ≥ 4

    def test_cancel_ok_by_target(self, tr):
        tr.begin((press(100.0, "atk"),))
        assert tr.cancel_window_open(100.05, "atk")
        assert tr.cancel_window_open(100.05, "skill1")
        assert not tr.cancel_window_open(100.05, "ult")    # 不在 cancel_into
        assert not tr.cancel_window_open(100.03, "atk")    # 窗口未开

    def test_no_window_after_done(self, tr):
        tr.begin((press(100.0, "atk"),))
        assert not tr.cancel_window_open(100.12, "atk")    # 招已打完 → 自由态


class TestCancelLink:
    """取消衔接: 下一招在上一招 cancel_from 帧按下 → query 切到最新招。"""

    def test_query_switches_to_latest_press(self, tr):
        # atk@100.0 → skill1@100.04 (gap=cancel_from 4帧=40ms)
        tr.begin((press(100.0, "atk"), press(100.04, "skill1")))
        q = tr.query(100.05)                        # skill1 已按下 10ms
        assert q.button == "skill1"
        assert q.phase is ActionPhase.STARTUP       # skill1.startup=3, frame=1

    def test_busy_through_whole_link(self, tr):
        tr.begin((press(100.0, "atk"), press(100.04, "skill1")))
        # skill1 total=7 帧 → 100.04+0.07=100.11 打完
        for t in (100.01, 100.045, 100.08, 100.109):
            assert tr.busy(t), f"t={t} 应处于 busy"
        assert not tr.busy(100.12)                  # 避开 70ms 精确边界(浮点取整)

    def test_pending_between_planned_steps(self, tr):
        # 造 pending: 上一招(atk total=10帧=100ms)打完后、下一招按下前的空窗
        tr.begin((press(100.0, "atk"), press(100.12, "ult")))
        q = tr.query(100.11)
        assert q.button is None and q.phase is ActionPhase.IDLE
        assert q.pending                            # 下一步(ult)在等待
        assert tr.busy(100.11)                      # pending 也算 busy(不许插队)

    def test_all_done_no_pending(self, tr):
        tr.begin((press(100.0, "atk"),))
        assert not tr.query(100.20).pending
        assert not tr.busy(100.20)

    def test_begin_rejects_empty(self, tr):
        with pytest.raises(ValueError):
            tr.begin(())

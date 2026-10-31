# -*- coding: utf-8 -*-
"""M4.7 单测 — ⑨ 博弈策略层: TradeScorer + IntentPolicy 决策梯 + Sequencer 接线。

决策梯覆盖: 被控停手 / 威胁侧移+强制dash / 残血替身 / 被墙角逃逸 /
逼墙压制(punish) / bait(事件分支→punish) / 常规进攻 / 拉开与接近。
状态全部经 StateBuilder 从 PerceptionResult 构造 —— 不手搓 64 维向量。
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from core.executor import Executor
from core.frame_data import FrameData
from core.injection import MockInjector
from core.keymap import Keymap
from core.perception import PerceptionResult
from core.policy import Action
from core.sequencer import Sequencer
from core.state import StateBuilder
from core.strategy import (DAMAGE_EST, IntentPolicy, TradeContext, TradeScorer,
                           combo_damage)


def state_of(**kw) -> np.ndarray:
    """PerceptionResult(**kw) → 64 维状态(与真机完全同一条组装路径)。"""
    defaults = dict(self_pos=(400.0, 500.0), enemy_pos=(800.0, 500.0),
                    enemy_visible=True, sub_ready_enemy=False,
                    event_remaining={"self_dash": 0.0},
                    wall_dist_self=1.0, wall_dist_enemy=1.0)
    defaults.update(kw)
    r = PerceptionResult(frame_id=0, ts=0.0, **defaults)
    return StateBuilder().build(r)


def make_policy(mini_fd=True):
    """(policy, sequencer, injector)。mini_fd=False 用生产帧表。"""
    fd = FrameData.load()
    inj = MockInjector(clock=time.monotonic)
    ex = Executor(inj, Keymap.load(), humanize=False)
    seqr = Sequencer(ex, fd)
    pol = IntentPolicy(fd=fd)
    pol.bind_sequencer(seqr)
    return pol, seqr, inj


# ═════════════════ TradeScorer ═════════════════
class TestTradeScorer:
    def test_combo_damage_sums(self):
        assert combo_damage(("skill1", "atk", "atk")) == pytest.approx(
            DAMAGE_EST["skill1"] + 2 * DAMAGE_EST["atk"])

    def test_sub_ready_increases_risk(self):
        t = TradeScorer()
        base = TradeContext(0.8, 0.6, enemy_sub_ready=False,
                            enemy_ult_threat=False, combo_damage=0.14)
        risky = TradeContext(0.8, 0.6, enemy_sub_ready=True,
                             enemy_ult_threat=False, combo_damage=0.14)
        assert t.score(risky) < t.score(base)

    def test_ult_threat_increases_risk_more(self):
        t = TradeScorer()
        base = TradeContext(0.8, 0.6, enemy_sub_ready=False,
                            enemy_ult_threat=False, combo_damage=0.14)
        ult = TradeContext(0.8, 0.6, enemy_sub_ready=False,
                           enemy_ult_threat=True, combo_damage=0.14)
        assert t.score(ult) < t.score(base)

    def test_hp_edge_encourages_engage(self):
        t = TradeScorer()
        losing = TradeContext(0.3, 0.9, enemy_sub_ready=False,
                              enemy_ult_threat=False, combo_damage=0.14)
        winning = TradeContext(0.9, 0.3, enemy_sub_ready=False,
                               enemy_ult_threat=False, combo_damage=0.14)
        assert t.score(winning) > t.score(losing)
        # 落后时同一套连段可能跌破阈值 → 更谨慎(阈值语义)
        assert t.should_engage(winning)
        assert not t.should_engage(losing, threshold=0.05)


# ═════════════════ 决策梯: 各梯级 ═════════════════
class TestIntentLadder:
    def test_cc_knockdown_stops_everything(self):
        pol, seqr, _ = make_policy()
        a = pol.decide(state_of(effects=frozenset({"cc"})))
        assert a.move == (0.0, 0.0) and a.button is None

    def test_imminent_threat_dodges_perpendicular(self):
        pol, seqr, _ = make_policy()
        # 弹道沿 -x 飞 → 侧移应为纯 y 向
        a = pol.decide(state_of(threat_active=True, threat_kind="projectile",
                                threat_ttc_s=0.2, threat_dir=(-1.0, 0.0)))
        assert abs(a.move[0]) < 1e-6 and abs(a.move[1]) > 0.9
        assert seqr.current_name == "dodge_dash"   # 强制 dash 已提交

    def test_low_hp_with_sub_ready_uses_sub(self):
        pol, seqr, _ = make_policy()
        a = pol.decide(state_of(hp_self=0.2, sub_ready_self=True))
        assert a.button == "sub"

    def test_cornered_escapes_toward_center(self):
        pol, seqr, _ = make_policy()
        # 自身贴左墙(100,500): 墙距70px → 70/500=0.14; 敌近(240,500) → 被墙角
        a = pol.decide(state_of(self_pos=(100.0, 500.0),
                                enemy_pos=(240.0, 500.0),
                                wall_dist_self=70.0 / 500.0))
        assert a.move[0] > 0.9                    # 朝中心(+x)逃
        assert seqr.current_name == "escape_dash"

    def test_enemy_cornered_triggers_wall_pin_punish(self):
        pol, seqr, _ = make_policy()
        # 敌贴右墙(1180,500): 墙距70px → 0.14; 距离380<420 → 逼墙压制
        a = pol.decide(state_of(self_pos=(800.0, 500.0),
                                enemy_pos=(1180.0, 500.0),
                                sub_ready_enemy=False,
                                wall_dist_enemy=70.0 / 500.0))
        assert seqr.current_name == "punish_sub"  # 替身不在 → 直接高伤
        # 站位: 卡在敌(1180)与中心之间的点(≈1070) → 自己(800)向 +x 移动
        assert a.move[0] > 0.9
        assert a.strength < 1.0                    # 压场半推

    def test_bait_submits_with_event_branch(self):
        pol, seqr, _ = make_policy()
        # 敌替身就绪 + 距离160 ∈ bait 窗 → bait_atk(带 enemy_sub 分支)
        a = pol.decide(state_of(enemy_pos=(560.0, 500.0),
                                sub_ready_enemy=True))
        assert seqr.current_name == "bait_atk"
        # ⑦ 事件分支: 感知到敌方替身闪 → 帧级切 punish
        assert seqr.on_events(("enemy_sub",)) is True
        assert seqr.current_name == "punish_sub"

    def test_engage_when_trade_positive(self):
        pol, seqr, _ = make_policy()
        # 敌替身不在 + 距离200<射程 + skill1 就绪 → 基础连段
        a = pol.decide(state_of(enemy_pos=(600.0, 500.0),
                                sub_ready_enemy=False))
        assert seqr.current_name == "engage_basic"

    def test_too_close_retreats(self):
        pol, seqr, _ = make_policy()
        # 距离 80 < RETREAT(130) → 反向拉开; 且替身不在CD不该 bait(距离<120)
        a = pol.decide(state_of(enemy_pos=(480.0, 500.0),
                                sub_ready_enemy=False))
        assert a.move[0] < -0.9

    def test_far_approaches(self):
        pol, seqr, _ = make_policy()
        a = pol.decide(state_of(enemy_pos=(1100.0, 500.0)))   # 距离700
        assert a.move[0] > 0.9


# ═════════════════ 兜底与接线 ═════════════════
class TestFallbackAndWiring:
    def test_unbound_policy_returns_dash_action(self):
        """未绑定 Sequencer → 躲避退化为单帧 Action(button='dash')。"""
        pol = IntentPolicy(fd=FrameData.load())
        a = pol.decide(state_of(threat_active=True, threat_kind="projectile",
                                threat_ttc_s=0.2, threat_dir=(-1.0, 0.0)))
        assert a.button == "dash"

    def test_action_progress_reaches_state_via_orchestrator(self):
        """Orchestrator 反馈链: action_progress/watchdog 进 meta(接口存在性)。"""
        from core.orchestrator import Orchestrator
        from core.detectors import MockFastDetector
        from core.perception import CVReader, Perception
        from core.screen_capture import SyntheticSource
        from core.synthetic import Scene
        pol, seqr, inj = make_policy()
        src = SyntheticSource(fps=60, scene=Scene())
        perc = Perception(MockFastDetector(), CVReader(Keymap.load()))
        orch = Orchestrator(src, perc, StateBuilder(), pol, seqr.executor,
                            sequencer=seqr)
        rpt = orch.run(ticks=6)
        assert rpt.ticks == 6
        assert inj.contacts_pressed() == set()      # 收尾零残留
        assert "submitted" in rpt.seq_stats

    def test_dash_press_starts_self_dash_timer(self):
        """自己 dash → self_dash 事件计时器起表(dash_cd 状态来源)。"""
        from core.orchestrator import Orchestrator
        from core.detectors import MockFastDetector
        from core.perception import CVReader, Perception
        from core.screen_capture import SyntheticSource
        from core.synthetic import Scene
        from core.policy import PolicyProvider

        class DashBot(PolicyProvider):
            name = "dashbot"

            def decide(self, state):
                return Action(button="dash")

        km = Keymap.load()
        inj = MockInjector(clock=time.monotonic)
        ex = Executor(inj, km, humanize=False)
        seqr = Sequencer(ex, FrameData.load())
        src = SyntheticSource(fps=60, scene=Scene())
        perc = Perception(MockFastDetector(), CVReader(km))
        orch = Orchestrator(src, perc, StateBuilder(), DashBot(), ex,
                            sequencer=seqr)
        orch.run(ticks=3)
        assert perc._events["self_dash"].running is True

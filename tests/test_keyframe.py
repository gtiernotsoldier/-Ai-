# -*- coding: utf-8 -*-
"""关键决策点提取的契约测试。

守的是四件事:
  ① 六路信号各自能被正确识别
  ② NMS 真的抑制住了「一次连段吐出二十个点」
  ③ 「无事发生」的量化不随合成数据退化
  ④ 布局下标全部从 StateLayout 推导, 没有一个写死的魔法数
"""
import numpy as np
import pytest

from core.constants import THREAT_TTC_NORM_S
from core.episode import Episode, EpisodeMeta
from core.keyframe import (DIST_FULL_PX, ImpactWeights, eventless_mask,
                           eventless_ratio, extract, extract_many)
from core.policy import AXIS_IDLE
from core.state import DEFAULT_LAYOUT

HP_S = DEFAULT_LAYOUT.index("resources", "hp_self")
HP_E = DEFAULT_LAYOUT.index("resources", "hp_enemy")
VIS = DEFAULT_LAYOUT.index("enemy_kin", "visible")
DIST = DEFAULT_LAYOUT.index("relative", "dist")
TTC = DEFAULT_LAYOUT.index("meta", "threat_ttc")

DT = 0.05
IDLE = (AXIS_IDLE, AXIS_IDLE, 0, 0)
ATK = (AXIS_IDLE, AXIS_IDLE, 1, 0)


def _state(*, hp_s=1.0, hp_e=1.0, dist_px=300.0, visible=True, ttc_s=0.0):
    s = np.full(64, 0.5, dtype=np.float32)
    s[HP_S] = hp_s
    s[HP_E] = hp_e
    s[VIS] = 1.0 if visible else 0.0
    s[DIST] = dist_px / DIST_FULL_PX
    s[TTC] = ttc_s / THREAT_TTC_NORM_S
    return s


def _pick(v, i, default):
    if v is None:
        return default
    if np.isscalar(v):
        return v
    return v[i]


def _make(n=40, *, hp_s=None, hp_e=None, dist=None, vis=None, ttc=None,
          action=None, result="timeout"):
    """按逐帧序列造一局。标量 = 全程恒定, 序列 = 逐帧取值。"""
    ep = Episode(EpisodeMeta(episode_id="t", started_at=0.0,
                             fps_nominal=20.0, result=result))
    for i in range(n):
        ep.append(
            i * DT, i,
            _state(hp_s=_pick(hp_s, i, 1.0), hp_e=_pick(hp_e, i, 1.0),
                   dist_px=_pick(dist, i, 300.0),
                   visible=_pick(vis, i, True),
                   ttc_s=_pick(ttc, i, 0.0)),
            True, _pick(action, i, IDLE), hold_idx=3, reward=0.0)
    ep.finish(result)
    return ep


def _idx(rep):
    return [e.idx for e in rep.events]


# ═════════════════════════ 六路信号 ═════════════════════════
class TestSignals:

    def test_hp_swing_detected(self):
        ep = _make(40, hp_e=[1.0] * 10 + [0.75] * 30)
        rep = extract(ep)
        assert 10 in _idx(rep)
        ev = next(e for e in rep.events if e.idx == 10)
        assert ev.reasons[0].startswith("hp")
        assert ev.impact >= 6.0 * 0.25          # hp 自身的贡献已够强

    def test_hp_swing_below_threshold_ignored(self):
        # 掉 0.02 血 → 0.12 分, 低于默认阈值 0.6
        ep = _make(40, hp_e=[1.0] * 10 + [0.98] * 30)
        assert _idx(extract(ep)) == []

    def test_engage_detected(self):
        ep = _make(40, action=[IDLE] * 10 + [ATK] * 30)
        rep = extract(ep)
        assert 10 in _idx(rep)
        assert "起手" in next(e for e in rep.events if e.idx == 10).reasons

    def test_disengage_detected(self):
        ep = _make(40, action=[ATK] * 10 + [IDLE] * 30)
        rep = extract(ep)
        assert 10 in _idx(rep)
        assert "收招" in next(e for e in rep.events if e.idx == 10).reasons

    def test_threat_appear_detected(self):
        ep = _make(40, ttc=[0.0] * 15 + [0.35] * 25)
        rep = extract(ep)
        assert 15 in _idx(rep)
        assert "威胁出现" in next(e for e in rep.events if e.idx == 15).reasons[0]

    def test_threat_persisting_emits_once(self):
        """持续存在的威胁只在该出现的那一帧算关键点, 不是每帧。"""
        ep = _make(60, ttc=[0.0] * 15 + [0.35] * 45)
        rep = extract(ep)
        assert sum(1 for e in rep.events if "威胁" in e.reasons[0]) == 1

    def test_visibility_flip_detected(self):
        ep = _make(40, vis=[True] * 12 + [False] * 28)
        rep = extract(ep)
        assert 12 in _idx(rep)
        assert "敌方消失" in next(e for e in rep.events if e.idx == 12).reasons

    def test_distance_band_cross_detected(self):
        # 500px(远) → 200px(中), 跨过 420 边界
        ep = _make(40, dist=[500.0] * 20 + [200.0] * 20)
        rep = extract(ep)
        assert 20 in _idx(rep)
        assert "距离跨带" in next(e for e in rep.events if e.idx == 20).reasons[0]

    def test_payoff_co_occurs_with_hp_drop(self):
        """掉血的那一帧应当同时被 hp 与回报落差两路信号命中。"""
        ep = _make(40, hp_e=[1.0] * 10 + [0.7] * 30)
        ev = next(e for e in extract(ep).events if e.idx == 10)
        assert any(r.startswith("回报落差") for r in ev.reasons)


# ═════════════════════════ NMS 与筛选 ═════════════════════════
class TestSelection:

    def test_nms_collapses_a_combo(self):
        """连续 6 帧每帧掉 0.12 血 —— 一次连段, 只能留一个点。"""
        hp = [1.0] * 10 + [1.0 - 0.12 * (k + 1) for k in range(6)] + [0.28] * 24
        rep = extract(_make(40, hp_e=hp), min_gap_s=0.4)
        hits = [i for i in _idx(rep) if 10 <= i <= 15]
        assert len(hits) == 1

    def test_min_gap_zero_keeps_all(self):
        hp = [1.0] * 10 + [1.0 - 0.12 * (k + 1) for k in range(6)] + [0.28] * 24
        rep = extract(_make(40, hp_e=hp), min_gap_s=0.0)
        assert len([i for i in _idx(rep) if 10 <= i <= 15]) == 6

    def test_threshold_filters_still_episode(self):
        rep = extract(_make(60))
        assert rep.events == []
        assert rep.ratio == 0.0

    def test_top_k_keeps_strongest(self):
        # idx=10 掉 0.1 血, idx=20 掉 0.5 血 —— 只留最强的一个
        ep = _make(40, hp_e=[1.0] * 10 + [0.9] * 10 + [0.4] * 20)
        rep = extract(ep, top_k=1, min_gap_s=0.0)
        assert len(rep.events) == 1
        assert rep.events[0].idx == 20

    def test_weights_are_overridable(self):
        ep = _make(40, vis=[True] * 12 + [False] * 28)
        weak = extract(ep, weights=ImpactWeights(flip=0.1), threshold=0.0)
        strong = extract(ep, weights=ImpactWeights(flip=20.0))
        assert "visibility" in weak.by_kind
        assert strong.top(1)[0].idx == 12

    def test_reasons_sorted_by_contribution(self):
        # 同时掉血 + 起手: hp 权重 6.0 > 起手 2.0, hp 应排第一
        ep = _make(40, hp_e=[1.0] * 10 + [0.6] * 30,
                   action=[IDLE] * 10 + [ATK] * 30)
        ev = next(e for e in extract(ep).events if e.idx == 10)
        assert ev.reasons[0].startswith("hp")
        assert "起手" in ev.reasons


# ═════════════════════════ 「无事发生」的量化 ═════════════════════════
class TestEventless:

    def test_still_episode_is_fully_eventless(self):
        ep = _make(60)
        assert eventless_ratio(ep) == pytest.approx(1.0)
        assert eventless_mask(ep).all()

    def test_always_changing_is_not_eventless(self):
        acts = [(AXIS_IDLE, AXIS_IDLE, i % 6 + 1, 0) for i in range(60)]
        ep = _make(60, action=acts)
        assert eventless_ratio(ep) < 0.05

    def test_hp_drop_breaks_eventlessness(self):
        ep = _make(60, hp_e=[1.0] * 30 + [0.5] * 30)
        m = eventless_mask(ep)
        assert not m[30]
        assert m[:30].all()

    def test_longest_lull_reported(self):
        ep = _make(60)
        rep = extract(ep)
        assert rep.longest_lull_s == pytest.approx(59 * DT, abs=0.06)

    def test_summary_has_all_keys(self):
        rep = extract(_make(40, hp_e=[1.0] * 10 + [0.6] * 30))
        s = rep.summary()
        for k in ("frames", "keyframes", "ratio", "eventless_ratio",
                  "longest_lull_s", "by_kind"):
            assert k in s
        assert s["frames"] == 40


# ═════════════════════════ 边界与导出 ═════════════════════════
class TestEdge:

    def test_empty_episode(self):
        ep = Episode(EpisodeMeta(episode_id="e", started_at=0.0))
        rep = extract(ep)
        assert rep.events == [] and rep.frame_count == 0
        assert eventless_ratio(ep) == 0.0

    def test_single_frame(self):
        ep = _make(1, action=[ATK])
        rep = extract(ep)
        assert rep.frame_count == 1
        assert isinstance(rep.events, list)

    def test_states_shape_and_content(self):
        hp = [1.0] * 10 + [0.5] * 30
        ep = _make(40, hp_e=hp)
        rep = extract(ep)
        st = rep.states()
        assert st.shape == (len(rep.events), 64)
        ev = next(e for e in rep.events if e.idx == 10)
        assert ev.state[HP_E] == pytest.approx(0.5, abs=1e-3)

    def test_to_records_is_json_ready(self):
        import json
        rep = extract(_make(40, hp_e=[1.0] * 10 + [0.6] * 30))
        json.dumps(rep.to_records())            # 不可序列化会直接抛

    def test_describe_readable(self):
        rep = extract(_make(40, hp_e=[1.0] * 10 + [0.6] * 30))
        line = rep.top(1)[0].describe()
        assert "s #" in line and "hp 跳变" in line

    def test_event_carries_episode_id(self):
        rep = extract(_make(40, hp_e=[1.0] * 10 + [0.6] * 30))
        assert rep.events[0].episode_id == "t"


# ═════════════════════════ 多局聚合 ═════════════════════════
class TestExtractMany:

    def test_merges_and_sorts(self):
        a = _make(40, hp_e=[1.0] * 10 + [0.9] * 30)      # 掉 0.1
        b = _make(40, hp_e=[1.0] * 10 + [0.4] * 30)      # 掉 0.6, 更强
        rep = extract_many([a, b])
        assert rep.frame_count == 80
        assert len(rep.events) == 2
        assert rep.events[0].impact > rep.events[1].impact
        assert sum(rep.by_kind.values()) == 2

    def test_per_episode_cap(self):
        a = _make(80, hp_e=[1.0] * 10 + [0.9] * 10 + [0.7] * 10 + [0.4] * 10
                  + [0.2] * 40)
        rep = extract_many([a], per_episode_cap=2)
        assert len(rep.events) == 2

    def test_empty_list(self):
        rep = extract_many([])
        assert rep.events == [] and rep.frame_count == 0
        assert rep.ratio == 0.0


# ═════════════════════════ 布局下标 ═════════════════════════
class TestLayoutIndex:

    def test_index_resolves_fields(self):
        assert DEFAULT_LAYOUT.index("resources", "hp_self") == 24
        assert DEFAULT_LAYOUT.index("resources", "hp_enemy") == 25
        assert DEFAULT_LAYOUT.index("meta", "threat_ttc") == 62

    def test_index_rejects_unknown(self):
        with pytest.raises(KeyError):
            DEFAULT_LAYOUT.index("resources", "hp_ghost")
        with pytest.raises(KeyError):
            DEFAULT_LAYOUT.index("nope", "x")

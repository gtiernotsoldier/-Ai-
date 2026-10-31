# -*- coding: utf-8 -*-
"""M4.7 单测 — ⑧ 检测器扩类: 威胁模型(弹道TTC/预警圈) + 墙体模型 + 状态回填。

核心语义:
  - TTC = 最近逼近法(非逐帧距离): 侧向飞过打不中的弹不算威胁
  - 墙: 检出的墙框优先; 无框才兜底到画面边界(人在场内→到最近边的距离)
  - effects 8 维 / flank / engage / dash_cd / threat_ttc+dir 真实回填
"""
from __future__ import annotations

import math

import numpy as np

from core.constants import (
    FRAME_H, FRAME_W, HITBOX_WARN_TTC_S, THREAT_TTC_NORM_S, WALL_DIST_NORM_PX,
)
from core.detectors import Detection, MockFastDetector
from core.perception import CVReader, Perception
from core.keymap import Keymap
from core.state import StateBuilder
from core.synthetic import HitboxWarn, Projectile, Scene
from core.screen_capture import SyntheticSource
from core.threat import ThreatTracker, WallModel


# ═════════════════ 弹道 TTC ═════════════════
class TestThreatTracker:
    def test_head_on_projectile_hits(self):
        """迎面飞来的弹: 两帧观测出速度 → TTC=距离/速度, 来向=飞行方向。"""
        tr = ThreatTracker()
        me = (400.0, 500.0)
        tr.update([Detection("projectile", 1000 - 8, 500 - 8, 16, 16, 0.95)],
                  me, 0.0)
        s = tr.update([Detection("projectile", 940 - 8, 500 - 8, 16, 16, 0.95)],
                      me, 0.1)
        assert s.active and s.kind == "projectile"
        assert s.ttc_s == pytest_approx(0.9, abs=0.05)   # 540px / 600px·s⁻¹
        assert s.dir_x == pytest_approx(-1.0, abs=1e-6)  # 向 -x 飞
        assert s.dir_y == pytest_approx(0.0, abs=1e-6)

    def test_sideways_projectile_misses(self):
        """侧向飞过、最小接近距离超过命中半径 → 不是威胁(不躲空枪)。"""
        tr = ThreatTracker()
        me = (400.0, 500.0)
        tr.update([Detection("projectile", 500 - 8, 100 - 8, 16, 16, 0.95)],
                  me, 0.0)
        s = tr.update([Detection("projectile", 440 - 8, 100 - 8, 16, 16, 0.95)],
                      me, 0.1)
        assert not s.active

    def test_stationary_effect_is_not_threat(self):
        """静止特效(速度<THREAT_MIN_SPEED)不参与 TTC。"""
        tr = ThreatTracker()
        me = (400.0, 500.0)
        tr.update([Detection("projectile", 450 - 8, 500 - 8, 16, 16, 0.95)],
                  me, 0.0)
        s = tr.update([Detection("projectile", 450 - 8, 500 - 8, 16, 16, 0.95)],
                      me, 0.1)
        assert not s.active

    def test_track_forgotten_after_ttl(self):
        """弹消失(检测断流)超过 TTL → 不再误报。"""
        tr = ThreatTracker()
        me = (400.0, 500.0)
        tr.update([Detection("projectile", 1000 - 8, 500 - 8, 16, 16, 0.95)],
                  me, 0.0)
        tr.update([Detection("projectile", 940 - 8, 500 - 8, 16, 16, 0.95)],
                  me, 0.1)
        s = tr.update([], me, 0.1 + ThreatTracker.TRACK_TTL_S + 0.05)
        assert not s.active


# ═════════════════ 预警圈 ═════════════════
class TestHitboxWarn:
    def test_warn_covering_self_is_threat(self):
        tr = ThreatTracker()
        me = (460.0, 500.0)                       # 圈心(500,450) r=90 → 罩住
        s = tr.update([Detection("hitbox_warn", 500 - 90, 450 - 90, 180, 180, 0.9)],
                      me, 0.0)
        assert s.active and s.kind == "hitbox_warn"
        assert s.ttc_s == pytest_approx(HITBOX_WARN_TTC_S)
        # 来向 = 圈心 → 自身方向(自身在圈心左下)
        assert s.dir_x < 0.0 and s.dir_y > 0.0

    def test_warn_far_away_is_not_threat(self):
        tr = ThreatTracker()
        s = tr.update([Detection("hitbox_warn", 500 - 90, 450 - 90, 180, 180, 0.9)],
                      (1000.0, 600.0), 0.0)
        assert not s.active


# ═════════════════ 墙体模型 ═════════════════
class TestWallModel:
    def test_fallback_bounds_distance_to_nearest_edge(self):
        wm = WallModel()
        wm.update([])
        pt, d = wm.nearest_wall_point((100.0, 360.0))
        assert pt == (30.0, 360.0) and d == pytest_approx(70.0)
        # 场中: 离上下边(330)比左右(610)近
        assert wm.nearest_wall_point((640.0, 360.0))[1] == pytest_approx(330.0)

    def test_detected_wall_box_wins_over_fallback(self):
        """墙框在场 → 以墙框为准, 兜底边界不参与竞争。"""
        wm = WallModel()
        wm.update([Detection("wall", 0, 0, 20, 720, 0.9)])
        pt, d = wm.nearest_wall_point((100.0, 360.0))
        assert pt == (20.0, 360.0) and d == pytest_approx(80.0)

    def test_no_walls_falls_back_each_frame(self):
        """上一帧有墙、本帧漏检 → 退回边界兜底(不留僵尸墙)。"""
        wm = WallModel()
        wm.update([Detection("wall", 0, 0, 20, 720, 0.9)])
        wm.update([])
        pt, d = wm.nearest_wall_point((100.0, 360.0))
        assert pt == (30.0, 360.0) and d == pytest_approx(70.0)


# ═════════════════ 感知→状态 回填 ═════════════════
def pytest_approx(x, abs=1e-5):
    import pytest
    return pytest.approx(x, abs=abs)


def _perception() -> Perception:
    return Perception(MockFastDetector(), CVReader(Keymap.load()))


class TestPerceptionThreatIntegration:
    def test_scene_threat_flows_to_result_and_state(self):
        scene = Scene(projectiles=(Projectile((1000.0, 500.0), (0.0, 0.0)),),
                      hitbox_warns=(HitboxWarn((500.0, 450.0), 90.0),))
        perc = _perception()
        src = SyntheticSource(fps=60, scene=scene)
        with src:
            r = perc.update(src.read())
        assert r.threat_active and r.threat_kind == "hitbox_warn"
        v = StateBuilder().build(r, 1, r.ts)
        assert v[62] == pytest_approx(HITBOX_WARN_TTC_S / THREAT_TTC_NORM_S)
        assert 0.0 < float(v[63]) <= 1.0          # 来向角已归一化

    def test_no_threat_zeroes_threat_dims(self):
        perc = _perception()
        src = SyntheticSource(fps=60, scene=Scene())
        with src:
            r = perc.update(src.read())
        assert not r.threat_active
        v = StateBuilder().build(r, 1, r.ts)
        assert v[62] == 0.0 and v[63] == 0.0

    def test_self_effects_reach_state(self):
        scene = Scene(effects={"i_frame", "cc", "sub_flash_self"})
        perc = _perception()
        src = SyntheticSource(fps=60, scene=scene)
        with src:
            r = perc.update(src.read())
        assert {"i_frame", "cc", "sub_flash_self"} <= r.effects
        v = StateBuilder().build(r, 1, r.ts)
        assert v[48] == 1.0                       # effects.i_frame
        assert v[50] == 1.0                       # effects.cc
        assert v[46] == 1.0                       # effects.sub_flash_self
        assert v[44] == 0.0                       # hit_flash 未出现

    def test_effect_sub_flags_and_fires_event(self):
        scene = Scene(effects={"sub_flash_enemy"})
        perc = _perception()
        src = SyntheticSource(fps=60, scene=scene)
        with src:
            r = perc.update(src.read())
        assert "sub_flash_enemy" in r.effects     # 当帧可见标志
        assert "enemy_sub" in r.events_fired      # 事件边沿(⑦)

    def test_wall_dist_normalized_in_state(self):
        scene = Scene(self_pos=(100.0, 500.0))    # 距左边界 70px
        perc = _perception()
        src = SyntheticSource(fps=60, scene=scene)
        with src:
            r = perc.update(src.read())
        assert r.wall_dist_self == pytest_approx(70.0 / WALL_DIST_NORM_PX)
        v = StateBuilder().build(r, 1, r.ts)
        assert v[54] == pytest_approx(70.0 / WALL_DIST_NORM_PX)

    def test_facing_flank_engage_filled(self):
        scene = Scene(self_pos=(1100.0, 350.0), enemy_pos=(800.0, 400.0),
                      facing_enemy=math.pi)       # 敌朝左, 我在敌右后 → 绕后
        perc = _perception()
        src = SyntheticSource(fps=60, scene=scene)
        with src:
            r = perc.update(src.read())
        assert r.facing_enemy == pytest_approx(math.pi)
        v = StateBuilder().build(r, 1, r.ts)
        assert v[21] == 1.0                       # relative.flank
        assert v[12] == pytest_approx((math.pi + math.pi) / (2 * math.pi))
        assert v[15] == 0.0                       # 敌静止 → engage=0

    def test_mock_detector_emits_new_classes(self):
        scene = Scene(projectiles=(Projectile((600.0, 300.0), (0.0, 0.0)),),
                      hitbox_warns=(HitboxWarn((700.0, 300.0), 80.0),),
                      walls=((0, 0, 20, 720),))
        src = SyntheticSource(fps=60, scene=scene)
        with src:
            dets = MockFastDetector().detect(src.read())
        classes = {d.cls for d in dets}
        assert {"projectile", "hitbox_warn", "wall"} <= classes

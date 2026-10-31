# -*- coding: utf-8 -*-
"""打磨清单剩余 P 级修复的回归测试。

P0-3: atk/secret/dash 三键可执行
P1 : 状态向量死维度回填 + enemy_kin.sub_ready 张冠李戴修复
P2#4: 敌人失踪 → 保留最后已知位置 + 置信度衰减
P2#5: resources.ult_cd_self 与 skill_cds.ult 量纲分离
① : 事件计时器生产路径注册 + 敌我替身就绪状态
⑤ : buttons 切片按钮反馈
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from core.constants import CD_TOTAL_S, EVENT_CD_S
from core.detectors import Detection, FastDetector
from core.executor import Executor
from core.injection import MockInjector
from core.keymap import Keymap
from core.perception import CVReader, Perception, PerceptionResult
from core.policy import Action, action_from_index
from core.state import StateBuilder
from core.synthetic import Scene
from core.screen_capture import SyntheticSource


# ═════════════════ P0-3: 三键存在且可执行 ═════════════════
class TestP0_3NewButtons:
    def test_action_buttons_complete(self):
        assert {"atk", "secret", "dash"} <= set(
            __import__("core.constants", fromlist=["ACTION_BUTTONS"]).ACTION_BUTTONS)

    def test_action_accepts_new_buttons(self):
        for b in ("atk", "secret", "dash"):
            Action(button=b)                      # 不再 raise

    def test_action_from_index_maps_new_buttons(self):
        assert action_from_index(14).button == "atk"
        assert action_from_index(15).button == "secret"
        assert action_from_index(16).button == "dash"

    def test_keymap_has_new_regions(self):
        km = Keymap.load()
        for b in ("atk", "secret", "dash"):
            cx, cy, r = km.circle(f"btn_{b}")     # 存在且可读
            assert 0 <= cx - r and cx + r <= 1280
            assert 0 <= cy - r and cy + r <= 720

    def test_executor_can_tap_new_buttons(self):
        km = Keymap.load()
        inj = MockInjector()
        ex = Executor(inj, km, clock=_VirtualClockForTests(), humanize=False)
        ex.play(ex.tap_macro("atk", hold_ms=30))
        km_cx, km_cy, _ = km.circle("btn_atk")
        downs = [e.event for e in inj.events if e.event.action == "d"]
        assert downs and (downs[0].x, downs[0].y) == (km_cx, km_cy)


class _VirtualClockForTests:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def wait(self, seconds, stop=None):
        self.t += seconds
        return False


# ═════════════════ P1: 张冠李戴修复 ═════════════════
class TestP1EnemySubReady:
    def test_enemy_kin_sub_ready_uses_enemy_side(self):
        r = PerceptionResult(frame_id=0, ts=0.0,
                             sub_ready_self=True, sub_ready_enemy=False)
        v = StateBuilder().build(r)
        assert v[14] == 0.0                      # 敌方未替身 → 0（旧bug填自己的1.0）
        r2 = PerceptionResult(frame_id=0, ts=0.0,
                              sub_ready_self=False, sub_ready_enemy=True)
        assert StateBuilder().build(r2)[14] == 1.0


# ═════════════════ P1: 死维度回填 ═════════════════
def fully_populated() -> tuple[PerceptionResult, dict]:
    r = PerceptionResult(
        frame_id=100, ts=12.0,
        self_pos=(640, 360), enemy_pos=(320, 180),
        self_vel=(200, 100), enemy_vel=(-100, 50),
        enemy_visible=True, enemy_pos_confidence=0.9, closing_speed=300.0,
        hp_self=0.8, hp_enemy=0.6, energy_self=0.7, energy_enemy=0.3,
        sub_ready_self=True, sub_ready_enemy=False,
        cds={"skill1": 0.3, "skill2": 0.2, "skill3": 0.1,
             "ult": 0.5, "sub": 0.0},
        event_remaining={"enemy_sub": 20.0, "self_sub": 10.0, "enemy_ult": 0.0},
    )
    fb = {"joy_x": 1.0, "joy_y": 0.0, "atk": True,
          "skill": False, "dash": False, "charged": False}
    return r, fb


class TestP1Backfill:
    def test_nonzero_dims_improved(self):
        r, fb = fully_populated()
        v = StateBuilder().build(r, 100, 12.0, buttons=fb)
        nz = int(np.count_nonzero(v))
        assert nz >= 36, f"回填后非零维度应 ≥36, 实为 {nz}（审计基线 24）"

    def test_buttons_feedback_wiring(self):
        r, fb = fully_populated()
        v = StateBuilder().build(r, 100, 12.0, buttons=fb)
        assert v[38] == pytest.approx(1.0)       # joy_x=1 → 0.5+0.5
        assert v[39] == pytest.approx(0.5)       # joy_y=0 → 中位
        assert v[40] == 1.0                      # atk
        assert v[41] == 0.0 and v[42] == 0.0 and v[43] == 0.0

    def test_buttons_default_neutral(self):
        r, _ = fully_populated()
        v = StateBuilder().build(r)              # 无反馈 → 摇杆中位, 按键全 0
        assert v[38] == 0.5 and v[39] == 0.5
        assert all(v[40 + i] == 0.0 for i in range(4))

    def test_closing_speed_and_approach(self):
        r, fb = fully_populated()
        v = StateBuilder().build(r, 100, 12.0, buttons=fb)
        assert v[20] == 1.0                      # closing>50 → approach=1
        assert v[22] == pytest.approx(0.5 + 300.0 / 2000.0)
        r2 = PerceptionResult(frame_id=0, ts=0.0, closing_speed=-100.0)
        v2 = StateBuilder().build(r2)
        assert v2[20] == 0.0 and v2[22] == pytest.approx(0.45)

    def test_threat_semantics_fixed(self):
        """旧语义: threat=对方残血; 新语义: 敌方大招就绪且近身。"""
        r, fb = fully_populated()                # ult ready + dist=367<400
        r.hp_enemy = 0.1                         # 残血但 ult 在 CD
        r.event_remaining["enemy_ult"] = 30.0
        v = StateBuilder().build(r, 100, 12.0, buttons=fb)
        assert v[23] == 0.0                      # 残血不再算威胁
        r.event_remaining["enemy_ult"] = 0.0     # ult 就绪 → 威胁
        assert StateBuilder().build(r, 100, 12.0, buttons=fb)[23] == 1.0

    def test_ult_cd_scales_differ_p2_5(self):
        r, fb = fully_populated()                # cds.ult=0.5
        v = StateBuilder().build(r, 100, 12.0, buttons=fb)
        assert v[35] == pytest.approx(0.5)                  # skill_cds: 按总CD归一
        assert v[30] == pytest.approx(0.5 * CD_TOTAL_S["ult"] / 60.0)  # 秒/60
        assert not v[30] == v[35] or CD_TOTAL_S["ult"] == 60.0

    def test_meta_event_remaining_backfill(self):
        r, fb = fully_populated()
        v = StateBuilder().build(r, 100, 12.0, buttons=fb)
        assert v[58] == pytest.approx(20.0 / EVENT_CD_S["enemy_sub"])
        assert v[59] == pytest.approx(10.0 / EVENT_CD_S["self_sub"])
        assert v[60] == pytest.approx(0.0)
        assert v[61] == pytest.approx(0.9)       # 敌方位置置信度

    def test_energy_diff_real_formula(self):
        r, fb = fully_populated()                # 0.7 vs 0.3
        v = StateBuilder().build(r, 100, 12.0, buttons=fb)
        assert v[19] == pytest.approx(0.5 + (0.7 - 0.3) / 2)   # 真差值


# ═════════════════ P2#4 + ①: 感知层 ═════════════════
class _GoneDetector(FastDetector):
    """第二帧起敌人消失（模拟被控/出屏）。"""

    def __init__(self):
        self.n = 0

    def detect(self, frame):
        self.n += 1
        if self.n == 1:
            return [Detection("enemy", 300, 200, 60, 60, 0.99),
                    Detection("self", 600, 400, 60, 60, 0.99)]
        return [Detection("self", 600, 400, 60, 60, 0.99)]


class TestPerceptionPolish:
    def test_event_timers_registered_by_default(self):
        p = Perception(_GoneDetector(), CVReader(Keymap.load()))
        assert set(p._events) == set(EVENT_CD_S)   # ①: 生产路径自动注册

    def test_enemy_lost_keeps_last_known_pos_and_decays(self):
        km = Keymap.load()
        p = Perception(_GoneDetector(), CVReader(km))
        src = SyntheticSource(fps=60, scene=Scene())
        with src:
            r1 = p.update(src.read())              # 可见
            assert r1.enemy_visible and r1.enemy_pos_confidence == 1.0
            pos_seen = r1.enemy_pos
            for _ in range(3):
                src.read()
                r = p.update(src.read())
        assert not r.enemy_visible
        assert r.enemy_pos == pos_seen             # 保留最后已知位置
        assert 0.0 < r.enemy_pos_confidence < 1.0  # 随时间衰减

    def test_effect_sub_starts_enemy_timer_and_flips_ready(self):
        from core.detectors import MockFastDetector
        km = Keymap.load()
        p = Perception(MockFastDetector(), CVReader(km))
        assert p._events["enemy_sub"].running is False
        scene = Scene(effects={"sub_flash_enemy"})
        src = SyntheticSource(fps=60, scene=scene)
        with src:
            r = p.update(src.read())
        assert p._events["enemy_sub"].running is True
        assert r.sub_ready_enemy is False          # 敌方替身进了 CD
        assert r.event_remaining["enemy_sub"] > EVENT_CD_S["enemy_sub"] - 1.0

    def test_closing_speed_positive_when_approaching(self):
        """自己朝敌动 → closing>0。"""
        class Mover(FastDetector):
            def detect(self, frame):
                t = frame.ts
                x = 300.0 + (t - t0) * 500.0       # 敌静止; 自己向右逼近
                return [Detection("enemy", 900, 400, 60, 60, 0.99),
                        Detection("self", x - 30, 400, 60, 60, 0.99)]
        t0 = time.monotonic()
        km = Keymap.load()
        p = Perception(Mover(), CVReader(km))
        src = SyntheticSource(fps=60, scene=Scene())
        with src:
            p.update(src.read())
            import time as _t
            _t.sleep(0.05)
            r = p.update(src.read())
        assert r.closing_speed > 0.0


# ═════════════════ ①+⑤: 闭环接线 ═════════════════
class TestOrchestratorWiring:
    def test_sub_press_starts_self_timer(self):
        from core.orchestrator import Orchestrator
        km = Keymap.load()
        src = SyntheticSource(fps=60, scene=Scene())
        perc = Perception(_GoneDetector(), CVReader(km))
        inj = MockInjector(clock=time.monotonic)
        ex = Executor(inj, km, humanize=False)
        orch = Orchestrator(src, perc, StateBuilder(),
                            _Fixed(Action(button="sub")), ex)
        orch.run(ticks=3)
        assert perc._events["self_sub"].running is True

    def test_buttons_feedback_reaches_state(self):
        """atk 动作在第 2 tick 的状态向量 buttons.atk=1。"""
        from core.orchestrator import Orchestrator
        km = Keymap.load()
        src = SyntheticSource(fps=60, scene=Scene())
        perc = Perception(_GoneDetector(), CVReader(km))
        inj = MockInjector(clock=time.monotonic)
        ex = Executor(inj, km, humanize=False)
        builder = StateBuilder()
        orch = Orchestrator(src, perc, builder,
                            _Fixed(Action(button="atk")), ex)
        seen: list[np.ndarray] = []
        orig = builder.build

        def spy(*a, **k):
            v = orig(*a, **k)
            seen.append(v.copy())
            return v

        builder.build = spy                        # type: ignore[method-assign]
        orch.run(ticks=3)
        assert len(seen) == 3
        assert seen[0][40] == 0.0                  # 首帧无反馈
        assert seen[1][40] == 1.0                  # 第2帧知道上一帧按了 atk
        assert seen[2][40] == 1.0


class _Fixed:
    name = "fixed"

    def __init__(self, action: Action):
        self.action = action

    def decide(self, state: np.ndarray) -> Action:
        return self.action

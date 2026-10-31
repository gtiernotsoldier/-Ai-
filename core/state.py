# -*- coding: utf-8 -*-
"""64 维状态向量组装（宪法: 状态组装预算 <1ms）。

布局单一事实来源 = StateLayout。决策层/训练层只认:
  - StateBuilder.build(...) -> np.float32(64,)
  - StateLayout.describe()   -> {维度区间: 说明}
全部取值归一化到 [0,1]（角度除以 2π）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.constants import STATE_DIM, FRAME_W, FRAME_H
from core.perception import PerceptionResult


@dataclass(frozen=True)
class Slice:
    name: str
    start: int
    end: int
    fields: tuple[str, ...]


class StateLayout:
    """64 维布局定义。修改此表 = 修改决策/训练接口契约, 需同步 onnx 导出。"""

    SLICES = (
        Slice("self_kin", 0, 8, ("x", "y", "vx", "vy", "facing", "alive", "moving", "dash_cd")),
        Slice("enemy_kin", 8, 16, ("x", "y", "vx", "vy", "facing", "visible", "sub_ready", "engage")),
        Slice("relative", 16, 24, ("dist", "angle", "hp_diff", "energy_diff", "approach", "flank", "closing_speed", "threat")),
        Slice("resources", 24, 32, ("hp_self", "hp_enemy", "energy_self", "energy_enemy",
                                    "sub_ready_self", "sub_ready_enemy", "ult_cd_self", "ult_cd_enemy")),
        Slice("skill_cds", 32, 38, ("skill1", "skill2", "skill3", "ult", "sub", "secret")),
        Slice("buttons", 38, 44, ("joy_x", "joy_y", "atk", "skill", "dash", "charged")),
        Slice("effects", 44, 52, ("hit_flash", "sub_flash_enemy", "sub_flash_self",
                                  "ult_cast", "i_frame", "knockdown", "cc", "super_armor")),
        Slice("meta", 52, 64, ("frame_id", "t", "slow_age", "fast_age",
                               "decision_gauge", "watchdog", "r6", "r7",
                               "r8", "r9", "r10", "r11")),
    )

    def __init__(self):
        total = sum(s.end - s.start for s in self.SLICES)
        assert total == STATE_DIM, f"布局总维度 {total} != {STATE_DIM}"
        covered = sorted(i for s in self.SLICES for i in range(s.start, s.end))
        assert covered == list(range(STATE_DIM)), "布局必须无重叠无遗漏覆盖 0..63"

    def describe(self) -> dict[str, tuple[str, ...]]:
        return {f"{s.name}[{s.start}:{s.end}]": s.fields for s in self.SLICES}


DEFAULT_LAYOUT = StateLayout()


class StateBuilder:
    """PerceptionResult(+可选上一帧按钮状态) → 64 维向量。"""

    def __init__(self, layout: StateLayout = DEFAULT_LAYOUT):
        self.layout = layout

    def build(self, r: PerceptionResult,
              frame_id: int = 0, now: float = 0.0) -> np.ndarray:
        v = np.zeros(STATE_DIM, dtype=np.float32)

        def put(name: str, idx: int, value: float) -> None:
            s = next(s for s in self.layout.SLICES if s.name == name)
            v[s.start + idx] = float(np.clip(value, 0.0, 1.0))

        _2pi = 2 * math.pi
        sx, sy = r.self_pos
        ex, ey = r.enemy_pos
        # self_kin（朝向信息由上层速度向量推算时占位 0.5）
        put("self_kin", 0, sx / FRAME_W); put("self_kin", 1, sy / FRAME_H)
        put("self_kin", 2, 0.5 + r.self_vel[0] / 2000); put("self_kin", 3, 0.5 + r.self_vel[1] / 2000)
        put("self_kin", 5, 1.0)
        put("self_kin", 6, 1.0 if abs(r.self_vel[0]) + abs(r.self_vel[1]) > 1 else 0.0)
        # enemy_kin
        put("enemy_kin", 0, ex / FRAME_W); put("enemy_kin", 1, ey / FRAME_H)
        put("enemy_kin", 2, 0.5 + r.enemy_vel[0] / 2000); put("enemy_kin", 3, 0.5 + r.enemy_vel[1] / 2000)
        put("enemy_kin", 5, 1.0 if r.enemy_visible else 0.0)
        put("enemy_kin", 6, 1.0 if r.sub_ready_self else 0.0)
        # relative
        dist = math.hypot(ex - sx, ey - sy)
        angle = (math.atan2(ey - sy, ex - sx) + math.pi) / _2pi
        put("relative", 0, dist / 1500.0)
        put("relative", 1, angle)
        put("relative", 2, 0.5 + (r.hp_self - r.hp_enemy) / 2)
        put("relative", 3, 0.5 + (r.energy_self - 0.5) / 2)
        put("relative", 6, 1.0 if dist < 200 else 0.0)
        put("relative", 7, 1.0 if r.hp_enemy < 0.2 else 0.0)
        # resources
        put("resources", 0, r.hp_self); put("resources", 1, r.hp_enemy)
        put("resources", 2, r.energy_self)
        put("resources", 4, 1.0 if r.sub_ready_self else 0.0)
        put("resources", 6, r.cds.get("ult", 0.0))
        # skill_cds
        for i, k in enumerate(("skill1", "skill2", "skill3", "ult", "sub")):
            put("skill_cds", i, r.cds.get(k, 0.0))
        # meta
        put("meta", 0, (frame_id % 1000) / 1000.0)
        put("meta", 1, now / 3600.0)

        return v

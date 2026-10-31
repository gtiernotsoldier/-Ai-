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
from typing import Optional

import numpy as np

from core.constants import (
    STATE_DIM, FRAME_W, FRAME_H,
    CD_TOTAL_S, EVENT_CD_S, THREAT_RANGE_PX,
    THREAT_TTC_NORM_S,
)
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
        Slice("meta", 52, 64, ("frame_id", "t", "wall_dist_self", "wall_dist_enemy",
                               "action_progress", "watchdog", "enemy_sub_s", "self_sub_s",
                               "enemy_ult_s", "enemy_pos_conf", "threat_ttc", "threat_dir")),
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
    """PerceptionResult(+可选上一帧按钮状态) → 64 维向量。

    buttons 反馈（打磨清单⑤/P1 死维度回填）: dict 键与 buttons 切片字段同名,
      joy_x/joy_y ∈ [-1,1]（内部转 [0,1]）, atk/skill/dash/charged 为 0/1。
      由 Orchestrator 从上一帧实际下发的 Action 构造 —— 策略层由此知道
      "我上一帧按了什么、蓄力蓄到哪了", 连段/取消/长按判断都依赖它。
    """

    def __init__(self, layout: StateLayout = DEFAULT_LAYOUT):
        self.layout = layout

    def build(self, r: PerceptionResult,
              frame_id: int = 0, now: float = 0.0,
              buttons: Optional[dict] = None) -> np.ndarray:
        """buttons: 上一帧执行反馈 dict（Orchestrator 构造）。

        除 buttons 切片的 6 键外还接受:
          action_progress  当前在飞招式的帧进度(0~1, 来自 ActionTracker)
          watchdog         上帧决策看门狗是否超时(0/1)
        """
        v = np.zeros(STATE_DIM, dtype=np.float32)
        fb = buttons or {}

        def put(name: str, idx: int, value: float) -> None:
            s = next(s for s in self.layout.SLICES if s.name == name)
            v[s.start + idx] = float(np.clip(value, 0.0, 1.0))

        _2pi = 2 * math.pi
        sx, sy = r.self_pos
        ex, ey = r.enemy_pos
        # self_kin（朝向: 运动时用速度方向估计; 静止/未知 → 0.5）
        put("self_kin", 0, sx / FRAME_W); put("self_kin", 1, sy / FRAME_H)
        put("self_kin", 2, 0.5 + r.self_vel[0] / 2000); put("self_kin", 3, 0.5 + r.self_vel[1] / 2000)
        spd = math.hypot(*r.self_vel)
        put("self_kin", 4, (math.atan2(r.self_vel[1], r.self_vel[0]) + math.pi) / _2pi
            if spd > 1.0 else 0.5)
        put("self_kin", 5, 1.0)
        put("self_kin", 6, 1.0 if abs(r.self_vel[0]) + abs(r.self_vel[1]) > 1 else 0.0)
        # ⑧ dash_cd: 自己按 dash 起表的事件计时器（真机无 dash HUD 区域, 同①推算）
        put("self_kin", 7, r.event_remaining.get("self_dash", 0.0)
            / EVENT_CD_S["self_dash"])
        # enemy_kin
        put("enemy_kin", 0, ex / FRAME_W); put("enemy_kin", 1, ey / FRAME_H)
        put("enemy_kin", 2, 0.5 + r.enemy_vel[0] / 2000); put("enemy_kin", 3, 0.5 + r.enemy_vel[1] / 2000)
        # ⑧ 敌方朝向(绕后/flank 感知来源); 未知 → 0.5
        put("enemy_kin", 4, (r.facing_enemy + math.pi) / _2pi
            if r.facing_enemy is not None else 0.5)
        put("enemy_kin", 5, 1.0 if r.enemy_visible else 0.0)
        # 修复打磨清单 P1「张冠李戴」: 敌方替身就绪必须填敌方侧,
        # 旧代码填的是 sub_ready_self → 策略会以为对手随时能替身
        put("enemy_kin", 6, 1.0 if r.sub_ready_enemy else 0.0)
        # ⑧ engage: 敌方正在向我逼近
        put("enemy_kin", 7, 1.0 if r.enemy_closing > 50.0 else 0.0)
        # relative
        dist = math.hypot(ex - sx, ey - sy)
        angle = (math.atan2(ey - sy, ex - sx) + math.pi) / _2pi
        put("relative", 0, dist / 1500.0)
        put("relative", 1, angle)
        put("relative", 2, 0.5 + (r.hp_self - r.hp_enemy) / 2)
        # 修复②: energy_diff 名不副实 —— 旧代码只用了自己的能量;
        # 敌方能量 HUD 不可见(恒0), 但公式先修正, 待检测器扩类后有数即用
        put("relative", 3, 0.5 + (r.energy_self - r.energy_enemy) / 2)
        # ④ 语义错位修复: approach=是否在接近, closing_speed=接近速度(px/s → 0.5±)
        put("relative", 4, 1.0 if r.closing_speed > 50.0 else 0.0)
        # ⑧ flank(绕后): 自己在敌方背后扇区 —— 背后 = (自-敌)与敌方朝向反向
        if r.facing_enemy is not None and dist > 1e-3:
            ahead = (math.cos(r.facing_enemy) * (sx - ex)
                     + math.sin(r.facing_enemy) * (sy - ey)) / dist
            put("relative", 5, 1.0 if ahead < 0.0 else 0.0)
        put("relative", 6, 0.5 + r.closing_speed / 2000.0)
        # ④ 语义错位修复: threat=敌方大招就绪且近身, 不再是"对方残血"
        enemy_ult_ready = r.event_remaining.get("enemy_ult", 0.0) <= 0.01
        put("relative", 7, 1.0 if (r.enemy_visible and enemy_ult_ready
                                   and dist < THREAT_RANGE_PX) else 0.0)
        put("resources", 0, r.hp_self); put("resources", 1, r.hp_enemy)
        put("resources", 2, r.energy_self)
        put("resources", 3, r.energy_enemy)          # HUD 不可见, 恒0, 公式先就位
        put("resources", 4, 1.0 if r.sub_ready_self else 0.0)
        put("resources", 5, 1.0 if r.sub_ready_enemy else 0.0)
        # P2#5: resources.ult_cd_self 改存 剩余秒数/60(绝对量纲, 1.0=60s),
        # 与 skill_cds.ult(按自身总CD归一化)不再同值重复
        put("resources", 6, r.cds.get("ult", 0.0) * CD_TOTAL_S["ult"] / 60.0)
        put("resources", 7, r.event_remaining.get("enemy_ult", 0.0)
            / EVENT_CD_S["enemy_ult"])
        # skill_cds: CVReader 扇形占比 = 剩余/自身总CD; secret 无 HUD 区域, 恒0
        for i, k in enumerate(("skill1", "skill2", "skill3", "ult", "sub")):
            put("skill_cds", i, r.cds.get(k, 0.0))
        # buttons 反馈（打磨清单⑤: 此前 6 维恒 0, 策略不知道自己上一帧按了什么）
        put("buttons", 0, 0.5 + fb.get("joy_x", 0.0) / 2)
        put("buttons", 1, 0.5 + fb.get("joy_y", 0.0) / 2)
        put("buttons", 2, 1.0 if fb.get("atk") else 0.0)
        put("buttons", 3, 1.0 if fb.get("skill") else 0.0)
        put("buttons", 4, 1.0 if fb.get("dash") else 0.0)
        put("buttons", 5, 1.0 if fb.get("charged") else 0.0)
        # effects 8 维（⑧ 检测器扩类后由感知层真实回填;
        # PerceptionResult.effects 的键名与切片字段名一一对齐）
        for i, name in enumerate(("hit_flash", "sub_flash_enemy", "sub_flash_self",
                                  "ult_cast", "i_frame", "knockdown", "cc", "super_armor")):
            put("effects", i, 1.0 if name in r.effects else 0.0)
        # meta
        put("meta", 0, (frame_id % 1000) / 1000.0)
        put("meta", 1, now / 3600.0)
        # ⑧ 墙距(归一化): 逼墙走位/被墙角判定/躲避选向的空间参照
        put("meta", 2, r.wall_dist_self)
        put("meta", 3, r.wall_dist_enemy)
        # 执行反馈: 当前在飞招式的帧进度 + 上帧看门狗是否超时
        put("meta", 4, fb.get("action_progress", 0.0))
        put("meta", 5, 1.0 if fb.get("watchdog") else 0.0)
        # 事件计时器剩余归一化 + 敌方位置置信度（打磨清单①）
        put("meta", 6, r.event_remaining.get("enemy_sub", 0.0)
            / EVENT_CD_S["enemy_sub"])
        put("meta", 7, r.event_remaining.get("self_sub", 0.0)
            / EVENT_CD_S["self_sub"])
        put("meta", 8, r.event_remaining.get("enemy_ult", 0.0)
            / EVENT_CD_S["enemy_ult"])
        put("meta", 9, r.enemy_pos_confidence)
        # ⑧ 最紧迫威胁: TTC(0=无威胁) + 来向角((θ+π)/2π, 无威胁恒 0)
        put("meta", 10, min(1.0, r.threat_ttc_s / THREAT_TTC_NORM_S)
            if r.threat_active else 0.0)
        if r.threat_active:
            put("meta", 11, (math.atan2(r.threat_dir[1], r.threat_dir[0])
                             + math.pi) / _2pi)

        return v

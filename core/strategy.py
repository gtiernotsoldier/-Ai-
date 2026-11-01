# -*- coding: utf-8 -*-
"""M4.7 — 博弈策略层 (strategy) —— 打磨清单施工⑨

20Hz 策略层升级版: 从 RulePolicy 的 4 条 if-else 升级为完整的博弈决策梯。

  TradeScorer   ② 资源置换评估器: 这套连段打出去值不值
                （期望伤害 − 被反打风险 + 血量优势项, 全部规则化可解释）
  IntentPolicy  博弈决策梯（只读 64 维状态, 输出单帧 Action + 序列意图）:
      0. 被控/倒地      → 停手（打不出任何东西）
      1. 即时威胁       → 垂直弹道侧移 + 强制 dash（保命级, force 打断连段）
      2. 残血替身       → 交替身（保命最优先, 同 RulePolicy）
      3. 被墙角         → 朝屏幕中心 dash 逃逸（墙角 = wall_dist 低 且敌近）
      4. 逼墙压制       → 敌方贴墙: 站位卡在「敌心→屏幕中心」方向上压场,
                          替身不在 CD 时直接上高伤 punish 连段
      5. bait/punish    → 敌方替身就绪: 低成本 bait 连段起手(带 enemy_sub 事件
                          分支 → 看到替身闪帧级切 punish, ⑦ 已在 Sequencer)
      6. 常规进攻       → trade score > 0 且在射程: 基础连段
      7. 默认走位       → 太近拉开 / 太远接近 / 中距半推压场

关键设计约束:
  - 只依赖 64 维状态向量(接口契约), 不偷看 PerceptionResult ——
    M6 换 OnnxPolicy 时同一状态可平替。
  - 躲避选向用「屏幕中心方向」近似「离墙方向」: 贴墙时中心方向必然背离墙,
    无需墙点进状态。逼墙站位同理(站到敌心与中心之间)。
  - 序列意图经 bind_sequencer() 注入的 Sequencer 提交; 未绑定(纯状态联调)时
    退化为单帧按钮 Action, 与 Orchestrator 旧路径兼容。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.frame_data import FrameData
from core.policy import Action, PolicyProvider, StateReader
from core.sequencer import Sequence, Sequencer, Step, combo_sequence
from core.state import REL_POS_FULL_PX


# ═════════════════════════ ② 资源置换评估 ═════════════════════════
# 每招期望伤害(占位标定值, 按血条比例; 真机标定后覆盖)
DAMAGE_EST: dict[str, float] = {
    "atk": 0.03, "skill1": 0.08, "skill2": 0.10, "skill3": 0.14, "ult": 0.25,
    "secret": 0.0, "dash": 0.0, "sub": 0.0,
}


def combo_damage(steps: tuple[str, ...]) -> float:
    """连段期望伤害 = 各招占位伤害求和（不计命中率, v1 简化）。"""
    return sum(DAMAGE_EST.get(b, 0.0) for b in steps)


@dataclass(frozen=True)
class TradeContext:
    """一次换血判断的全部上下文（全部来自状态向量）。"""
    hp_self: float
    hp_enemy: float
    enemy_sub_ready: bool      # 对面还能替身 → 打完可能白打/被反打
    enemy_ult_threat: bool     # 敌方大招就绪且近身(relative.threat=1)
    combo_damage: float        # 这套连段的期望伤害


class TradeScorer:
    """换血评分: score = 期望伤害 − 风险 + 血量优势项。

    score > threshold 才值得打。风险放大器:
      敌方替身就绪 ×1.6（打完可能被替身闪反打, 或伤害被闪掉）
      敌方大招威胁 ×2.0（换血换不过大招）
    血量优势项: (hp_self − hp_enemy) × 0.10 —— 领先时更敢打, 落后时更谨慎。
    """

    RISK_BASE = 0.06
    HP_ADV_WEIGHT = 0.10

    def score(self, ctx: TradeContext) -> float:
        risk = self.RISK_BASE
        if ctx.enemy_sub_ready:
            risk *= 1.6
        if ctx.enemy_ult_threat:
            risk *= 2.0
        hp_edge = ctx.hp_self - ctx.hp_enemy
        return ctx.combo_damage - risk + hp_edge * self.HP_ADV_WEIGHT

    def should_engage(self, ctx: TradeContext, threshold: float = 0.0) -> bool:
        return self.score(ctx) > threshold


# ═════════════════════════ ⑨ 博弈决策梯 ═════════════════════════
class IntentPolicy(PolicyProvider):
    """意图式规则策略: 走位/躲避走 Action, 连段/bait 走 Sequencer 序列意图。"""

    name = "intent"

    # ── 距离/范围（px）──
    ENGAGE_RANGE_PX = 320.0      # 技能/普攻射程
    PRESSURE_RANGE_PX = 420.0    # 压制范围（再远就先接近）
    RETREAT_RANGE_PX = 130.0     # 太近拉开
    BAIT_MIN_PX = 120.0          # bait 起手距离窗
    BAIT_MAX_PX = 380.0
    PIN_OFFSET_PX = 110.0        # 逼墙站位: 敌心→屏幕中心方向上的偏移
    # ── 阈值 ──
    DODGE_TTC_S = 0.40           # 威胁 0.4s 内命中 → 紧急侧移+dash
    CORNER_WALL_NORM = 0.20      # wall_dist < 0.2(≈100px) 视为贴墙
    CORNER_ENEMY_PX = 260.0      # 贴墙且敌近 → 被墙角, 逃逸
    BAIT_COOLDOWN_S = 2.5        # bait 起手冷却(防刷屏, Sequencer busy 亦是节流)
    SLOW_WALK_STRENGTH = 0.55    # 压场半推(④: 慢速压场需要 strength<1)
    ENGAGE_TRADE_THRESHOLD = 0.0

    def __init__(self, reader: StateReader | None = None,
                 fd: FrameData | None = None,
                 clock: Callable[[], float] | None = None):
        self.r = reader or StateReader()
        self.fd = fd
        self.trader = TradeScorer()
        self._seqr: Sequencer | None = None
        self._clock = clock or time.monotonic
        self._last_bait = -1e9

    # ── Orchestrator 自动装配入口(hasattr 约定) ──
    def bind_sequencer(self, seqr: Sequencer) -> None:
        self._seqr = seqr
        if self.fd is None:
            self.fd = seqr.fd

    # ── 工具 ──
    def _norm(self, dx: float, dy: float) -> tuple[float, float]:
        n = math.hypot(dx, dy)
        return (dx / n, dy / n) if n > 1e-6 else (0.0, 0.0)

    def _toward_center(self, from_xy: tuple[float, float]) -> tuple[float, float]:
        """指向屏幕中心的方向(归一化) —— 贴墙时它必然背离墙。

        ⚠️ 契约 v3 起自机绝对坐标已不进状态, 本函数仅保留给历史调用;
        新代码请用 _away_from_wall(state)。
        """
        return self._norm(0.5 - from_xy[0], 0.5 - from_xy[1])

    def _away_from_wall(self, state: np.ndarray) -> tuple[float, float]:
        """远离最近墙的方向（v3: self_kin.wall_dx/dy 直接给出墙向量）。

        贴墙时这是唯一正确的逃逸方向; 比旧的「朝屏幕中心」更准 ——
        镜头滚动时屏幕中心并不等于场地中心, 而墙向量始终成立。
        """
        wx = (self.r(state, "self_kin", "wall_dx") - 0.5) * 2
        wy = (self.r(state, "self_kin", "wall_dy") - 0.5) * 2
        return self._norm(-wx, -wy)

    def _submit(self, name: str, steps: tuple[str, ...],
                branches: tuple[tuple[str, str], ...] = (),
                force: bool = False) -> bool:
        """序列意图: 有 Sequencer → 编译提交; 无 → False(调用方退化)。"""
        if self._seqr is None:
            return False
        seq = (combo_sequence(self.fd, name, branches)
               if self.fd is not None and name in self.fd.combo_names
               else Sequence(name, tuple(Step(b) for b in steps), branches))
        return self._seqr.submit(seq, force=force)

    # ── 主决策 ──
    def decide(self, state: np.ndarray) -> Action:
        r = self.r
        # v3: 敌方位移直接相对自机（px）, 不再由两侧绝对坐标相减
        erel = ((r(state, "enemy_kin", "dx") - 0.5) * 2 * REL_POS_FULL_PX,
                (r(state, "enemy_kin", "dy") - 0.5) * 2 * REL_POS_FULL_PX)
        dist = r(state, "relative", "dist") * 1500.0
        enemy_visible = r(state, "enemy_kin", "visible") > 0.5
        hp_self = r(state, "resources", "hp_self")
        sub_ready = r(state, "resources", "sub_ready_self") > 0.5
        enemy_sub_ready = r(state, "enemy_kin", "sub_ready") > 0.5
        ult_threat = r(state, "relative", "threat") > 0.5
        dash_cd = r(state, "self_kin", "dash_cd")
        cc = (r(state, "effects", "cc") > 0.5
              or r(state, "effects", "knockdown") > 0.5)
        wall_self = r(state, "meta", "wall_dist_self")
        wall_enemy = r(state, "meta", "wall_dist_enemy")
        threat_ttc = r(state, "meta", "threat_ttc")
        threat_dir_norm = r(state, "meta", "threat_dir")

        # 0) 被控/倒地 → 停手
        if cc:
            return Action(move=(0.0, 0.0))

        # 1) 即时威胁 → 垂直弹道侧移 + 强制 dash(保命级)
        if threat_ttc > 0.0 and threat_ttc < self.DODGE_TTC_S:
            theta = threat_dir_norm * 2 * math.pi - math.pi    # 来向角
            tx, ty = math.cos(theta), math.sin(theta)
            p1, p2 = (-ty, tx), (ty, -tx)                      # 两个垂直向
            c = self._away_from_wall(state)
            if c == (0.0, 0.0):
                c = self._norm(-erel[0], -erel[1])             # 无墙信息 → 往敌反向
            perp = p1 if p1[0] * c[0] + p1[1] * c[1] >= p2[0] * c[0] + p2[1] * c[1] else p2
            if self._seqr is not None and dash_cd <= 0.05 \
                    and self._seqr.current_name != "dodge_dash":
                # dash 是序列步骤; force 打断在飞连段 —— 保命优先
                self._submit("dodge_dash", ("dash",), force=True)
            elif self._seqr is None:
                return Action(move=perp, button="dash" if dash_cd <= 0.05 else None)
            return Action(move=perp)

        # 2) 残血 + 替身可用 → 替身（保命最优先, 同 RulePolicy）
        if hp_self < 0.30 and sub_ready:
            return Action(button="sub")

        # 3) 被墙角 → 背离最近墙 dash 逃逸
        if (wall_self < self.CORNER_WALL_NORM
                and enemy_visible and dist < self.CORNER_ENEMY_PX):
            esc = self._away_from_wall(state)
            if esc == (0.0, 0.0):
                esc = self._norm(-erel[0], -erel[1])
            if self._seqr is not None and dash_cd <= 0.05 \
                    and self._seqr.current_name != "escape_dash":
                self._submit("escape_dash", ("dash",), force=True)
            elif self._seqr is None:
                return Action(move=esc, button="dash" if dash_cd <= 0.05 else None)
            return Action(move=esc)

        if not enemy_visible:
            # 敌方失踪: 背离墙角慢慢回到场地内侧(不空追, 等置信度恢复)
            c = self._away_from_wall(state)
            return Action(move=c, strength=self.SLOW_WALK_STRENGTH) \
                if c != (0.0, 0.0) else Action(move=(0.0, 0.0))

        # 4) 逼墙压制: 敌贴墙 → 卡在「自机→敌」线上, 到 PIN_OFFSET 停住;
        #    替身不在则上高伤 punish。
        enemy_cornered = wall_enemy < self.CORNER_WALL_NORM
        if enemy_cornered and dist < self.PRESSURE_RANGE_PX:
            if not enemy_sub_ready:
                # 对面替身不在 CD: 打不跑, 直接高伤 punish
                self._submit("punish_sub", ("skill2", "atk", "atk", "skill3"))
            mv = self._norm(*erel) if dist > self.PIN_OFFSET_PX else (0.0, 0.0)
            return Action(move=mv, strength=self.SLOW_WALK_STRENGTH)

        # 5) bait/punish: 敌方替身就绪 → 低成本起手逼它交(事件分支见 Sequencer)
        in_bait_window = self.BAIT_MIN_PX <= dist <= self.BAIT_MAX_PX
        if enemy_sub_ready and in_bait_window \
                and self._clock() - self._last_bait >= self.BAIT_COOLDOWN_S:
            if self._submit("bait_atk", ("atk", "atk"),
                            branches=(("enemy_sub", "punish_sub"),)):
                self._last_bait = self._clock()
            mv = self._norm(*erel)
            return Action(move=mv, strength=self.SLOW_WALK_STRENGTH)

        # 5.5) 太近拉开: 敌在我方近身圈内且不在压墙/连段意图中 → 拉开身位
        #      (必须在进攻梯级之前, 否则被 dist<射程 永久遮蔽)
        if dist < self.RETREAT_RANGE_PX:
            mv = self._norm(-erel[0], -erel[1])
            return Action(move=mv)

        # 6) 常规进攻: trade score > 0 且在射程
        if dist < self.ENGAGE_RANGE_PX:
            steps = ("skill1", "atk", "atk")
            ctx = TradeContext(hp_self=hp_self, hp_enemy=r(state, "resources", "hp_enemy"),
                               enemy_sub_ready=enemy_sub_ready,
                               enemy_ult_threat=ult_threat,
                               combo_damage=combo_damage(steps))
            skill1_cd = r(state, "skill_cds", "skill1")
            if self.trader.should_engage(ctx, self.ENGAGE_TRADE_THRESHOLD) \
                    and skill1_cd <= 0.01:
                self._submit("engage_basic", steps)
            # 边打边微调贴住(连段期间摇杆继续由持久流承载)
            mv = self._norm(*erel)
            return Action(move=mv, strength=self.SLOW_WALK_STRENGTH)

        # 7) 默认走位: 中距接近 / 远距全速接近
        mv = self._norm(*erel)
        return Action(move=mv, strength=self.SLOW_WALK_STRENGTH) \
            if dist < self.PRESSURE_RANGE_PX else Action(move=mv)

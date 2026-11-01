# -*- coding: utf-8 -*-
"""奖励函数 —— 直接抄 Shūkai（arXiv:2406.01103, ICML'24）。

腾讯在《火影忍者》手游上线的做法简单到反直觉:

    r_t = (HP_self[t+1] − HP_self[t]) + (HP_enemy[t] − HP_enemy[t+1])
    终局那一步 × 7

没有距离奖励、没有技能命中奖励、没有时间惩罚。血量差本身就是格斗游戏
最完整的信用分配 —— 打中了就掉血, 打空就不掉, 不需要人再教一遍。

论文原式第二项写成了 (HP_t^oppo − HP_t^oppo)（恒 0, 明显是笔误）,
这里按语义实现为「敌方掉血为正」。

三种人格（balanced / cautious / aggressive）不是改网络, 是改两个权重 ——
同一套策略结构可以长出三种风格的陪练。
"""
from __future__ import annotations

from dataclasses import dataclass

from core.state import DEFAULT_LAYOUT

# 终局放大系数（Shūkai 原值）
TERMINAL_MULT = 7.0


def _hp_indices(layout=DEFAULT_LAYOUT) -> tuple[int, int]:
    s = next(s for s in layout.SLICES if s.name == "resources")
    return (s.start + s.fields.index("hp_self"),
            s.start + s.fields.index("hp_enemy"))


@dataclass(frozen=True)
class Personality:
    """人格 = 对「自己掉血」和「敌人掉血」的相对敏感度。"""
    name: str
    self_damage_weight: float = 1.0     # 自己掉血记负, 权重越大越怕
    enemy_damage_weight: float = 1.0    # 敌人掉血记正, 权重越大越凶

    def __post_init__(self) -> None:
        if self.self_damage_weight <= 0 or self.enemy_damage_weight <= 0:
            raise ValueError("人格权重必须为正")


BALANCED = Personality("balanced", 1.0, 1.0)
CAUTIOUS = Personality("cautious", 1.5, 0.9)      # 防守反击: 更怕挨打
AGGRESSIVE = Personality("aggressive", 0.7, 1.2)  # 抢攻: 更看重输出

PERSONALITIES: dict[str, Personality] = {
    p.name: p for p in (BALANCED, CAUTIOUS, AGGRESSIVE)
}


def hp_delta(hp_self_prev: float, hp_self_now: float,
             hp_enemy_prev: float, hp_enemy_now: float) -> float:
    """原始血量差（未加权, 未放大）。全部为 [0,1] 归一化血量。"""
    return (hp_self_now - hp_self_prev) + (hp_enemy_prev - hp_enemy_now)


def step_reward(hp_self_prev: float, hp_self_now: float,
                hp_enemy_prev: float, hp_enemy_now: float,
                *, terminal: bool = False,
                personality: Personality = BALANCED,
                terminal_mult: float = TERMINAL_MULT) -> float:
    """单步奖励。terminal=True 时整步 ×7（胜负由血量差承载, 不另设胜负奖励）。"""
    r = (personality.self_damage_weight * (hp_self_now - hp_self_prev)
         + personality.enemy_damage_weight * (hp_enemy_prev - hp_enemy_now))
    return r * (terminal_mult if terminal else 1.0)


class HpReward:
    """有状态的逐步奖励器: 记住上一帧血量, 每 tick 给一个标量。

    训练循环里直接 `reward = tracker.update(state, terminal=done)` 即可;
    血量下标从 StateLayout 推导（单一事实来源）, 不手写魔法数字。
    """

    def __init__(self, personality: Personality = BALANCED,
                 terminal_mult: float = TERMINAL_MULT,
                 hp_index: tuple[int, int] | None = None):
        self.personality = personality
        self.terminal_mult = terminal_mult
        self._idx_self, self._idx_enemy = hp_index or _hp_indices()
        self._prev: tuple[float, float] | None = None

    def reset(self) -> None:
        self._prev = None

    def update(self, state, *, terminal: bool = False) -> float:
        hp_self = float(state[self._idx_self])
        hp_enemy = float(state[self._idx_enemy])
        if self._prev is None:
            self._prev = (hp_self, hp_enemy)
            return 0.0
        r = step_reward(self._prev[0], hp_self, self._prev[1], hp_enemy,
                        terminal=terminal, personality=self.personality,
                        terminal_mult=self.terminal_mult)
        self._prev = (hp_self, hp_enemy)
        return r

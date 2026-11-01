# -*- coding: utf-8 -*-
"""难度档位 —— 抄 Shūkai 的「预测延迟帧数 × 动作执行概率」（arXiv:2406.01103）。

腾讯在《火影忍者》手游上线的三档难度不是靠改策略强度, 而是靠改**人的生理约束**:
  beginner     反应更慢 + 有一定概率漏操作
  intermediate 介于两者之间
  advanced     接近人类高手（120ms 反应 / 几乎不失手）

这样做的好处: 同一套策略权重换档即变难度, 不需要训练三个模型;
而且高段位档天然保留拟人化, 不会变成帧完美机器人（宪法红线 4）。

对应 Shūkai 的部署 Elo 区间: 0-600 / 600-999 / 999-1299。
"""
from __future__ import annotations

from dataclasses import dataclass

from core.constants import HUMANIZE_MISS_RATE, HUMANIZE_REACTION_MS


@dataclass(frozen=True)
class HumanProfile:
    """一档「人类操作者」的参数。"""
    name: str
    delay_ms: float                    # 反应延迟（每条约 1 帧=16.7ms 的整数倍更真实）
    exec_prob: float = 1.0             # 动作真的被执行的概率（走神/手滑）
    miss_rate: float = HUMANIZE_MISS_RATE   # 坐标级失误扰动概率

    def __post_init__(self) -> None:
        if self.delay_ms < 0:
            raise ValueError(f"delay_ms 不可为负: {self.delay_ms}")
        if not 0.0 <= self.exec_prob <= 1.0:
            raise ValueError(f"exec_prob 需在 [0,1]: {self.exec_prob}")
        if not 0.0 <= self.miss_rate <= 1.0:
            raise ValueError(f"miss_rate 需在 [0,1]: {self.miss_rate}")


# advanced 与历史行为完全一致（120ms / 1%），保证换档不改变默认表现
ADVANCED = HumanProfile("advanced", delay_ms=float(HUMANIZE_REACTION_MS),
                        exec_prob=1.0, miss_rate=HUMANIZE_MISS_RATE)
INTERMEDIATE = HumanProfile("intermediate", delay_ms=160.0,
                            exec_prob=0.90, miss_rate=HUMANIZE_MISS_RATE)
BEGINNER = HumanProfile("beginner", delay_ms=220.0,
                        exec_prob=0.75, miss_rate=HUMANIZE_MISS_RATE)

DEFAULT_PROFILE = ADVANCED

PROFILES: dict[str, HumanProfile] = {
    p.name: p for p in (BEGINNER, INTERMEDIATE, ADVANCED)
}


def get_profile(name: str) -> HumanProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"未知难度档: {name!r}（可选 {sorted(PROFILES)}）") from None

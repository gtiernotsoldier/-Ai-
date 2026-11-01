# -*- coding: utf-8 -*-
"""core/keyframe.py — 关键决策点提取（纯离线, 零训练）。

一局 3200 帧里真正改变走向的可能只有 60~120 帧, 其余是「重复操作」:
摇杆方向没变、血量没变、威胁没变、距离在同一个带里抖动。

本模块把「去哪摸索」从「怎么摸索」里拆出来:

    这里做的事      用可计算的冲击力度量筛出候选关键点 —— 规则, 不需要任何数据
    RL/DAgger 做事 只在这些点上探索 / 请在旁的人纠正  —— 搜索空间降一个数量级

这个拆分很重要: 「哪些帧值得学」本身不需要智能, 「在这些帧上怎么做更好」才需要。
先筛后学, 3 分钟的录像变成 6 秒的重点片段, 人的注意力才够用。

六路信号, 全部从 episode 数组直接算出, 不需要额外采集:

    ① hp_swing   血量跳变          「发生了什么」的唯一硬证据
    ② payoff     回报落差 |ΔG|     事后视角, 就是你说的对局结束后的蒙特卡洛回报
    ③ engage     idle → 非 idle    起手
    ④ disengage  非 idle → idle    收招
    ⑤ threat     威胁从无到有      必须反应的瞬间
    ⑥ flip       可见性翻转 / 距离跨越交战带

打分后做时间维 NMS: 相邻关键点间隔 ≥ min_gap_s。否则一次 2 秒的连段会吐出
二十个点, 等于没筛。

用法::

    from core.keyframe import extract, eventless_ratio

    rep = extract(ep, terminal_bonus=None)     # None = 按 ep.result 自动推
    print(rep.ratio, rep.by_kind)
    for ev in rep.top(20):
        print(ev.describe())
    for ev in rep.events:                       # start-state curriculum
        reopen_from(ev.state)                   # 从这个 64 维向量重开
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.constants import THREAT_TTC_NORM_S
from core.episode import Episode, discounted_returns
from core.policy import AXIS_IDLE
from core.state import DEFAULT_LAYOUT

_L = DEFAULT_LAYOUT
I_HP_SELF = _L.index("resources", "hp_self")
I_HP_ENEMY = _L.index("resources", "hp_enemy")
I_VISIBLE = _L.index("enemy_kin", "visible")
I_DIST = _L.index("relative", "dist")
I_TTC = _L.index("meta", "threat_ttc")

# relative.dist 的归一化满量程（state.py: d / 1500）
DIST_FULL_PX = 1500.0
# 交战带边界（与 Strategy 的 RETREAT_RANGE_PX / PRESSURE_RANGE_PX 同口径）
BAND_EDGES_PX = (130.0, 420.0)
# payoff 的绝对尺度: 一次技能命中的血量量级。
# 刻意不用「本局分位数」归一化 —— 一局几乎无伤时, 分位数逼近 0, 会把 0.02
# 的掉血放大成满格信号, 于是满屏都是「关键点」, 等于没筛。
PAYOFF_SCALE_HP = 0.25

_RESULT_BONUS = {"win": 1.0, "lose": -1.0}


# ═════════════════════════ 权重与结果类型 ═════════════════════════
@dataclass(frozen=True)
class ImpactWeights:
    """六路信号的权重。默认值按「一次技能命中 ≈ 1 分」标定, 可直接调。"""
    hp_swing: float = 6.0        # × 血量差(0~1), 连段命中约 0.25 → 1.5
    # payoff 与 hp_swing 高度相关（G 主要由血量构成）, 权重必须明显低于它,
    # 否则同一件事被记两遍。它真正独占的贡献在【终局】: terminal_bonus 只走
    # 这一路, 于是「最后一击 / 被终结」会额外突出。
    payoff: float = 1.2          # × 归一化回报落差(0~2)
    engage: float = 2.0          # 起手是布尔量
    disengage: float = 1.2        # 收招信息量低于起手
    threat: float = 1.5
    flip: float = 1.5
    distance: float = 0.8


@dataclass(frozen=True)
class KeyEvent:
    """一个关键决策点。"""
    idx: int
    ts: float
    impact: float
    reasons: tuple[str, ...]
    state: np.ndarray = field(repr=False)     # float16(64,) —— 可直接当重开状态
    episode_id: str = ""

    def describe(self) -> str:
        tag = f"{self.episode_id} " if self.episode_id else ""
        return (f"[{tag}{self.ts:7.2f}s #{self.idx:4d}] {self.impact:5.2f}  "
                + " + ".join(self.reasons))


@dataclass
class KeyframeReport:
    """一局的关键点提取结果。"""
    events: list[KeyEvent]
    frame_count: int
    by_kind: dict[str, int]
    eventless_ratio: float
    longest_lull_s: float

    @property
    def ratio(self) -> float:
        """关键点占全部帧的比例 —— 通常 1%~5%。"""
        return len(self.events) / self.frame_count if self.frame_count else 0.0

    def top(self, k: int) -> list[KeyEvent]:
        return sorted(self.events, key=lambda e: -e.impact)[:k]

    def states(self) -> np.ndarray:
        """所有关键点的状态, 形状 (K, 64) —— start-state curriculum 的原料。"""
        if not self.events:
            return np.zeros((0, 64), dtype=np.float16)
        return np.stack([e.state for e in self.events])

    def to_records(self) -> list[dict]:
        """导出成人审清单（JSONL 一行一条）。"""
        return [{"episode": e.episode_id, "idx": e.idx, "ts": round(e.ts, 3),
                 "impact": round(e.impact, 3), "reasons": list(e.reasons)}
                for e in self.events]

    def summary(self) -> dict:
        return {
            "frames": self.frame_count,
            "keyframes": len(self.events),
            "ratio": round(self.ratio, 4),
            "eventless_ratio": round(self.eventless_ratio, 3),
            "longest_lull_s": round(self.longest_lull_s, 2),
            "by_kind": self.by_kind,
        }


# ═════════════════════════ 信号 ═════════════════════════
def hp_reward(ep: Episode) -> np.ndarray:
    """从状态重算 HP 差回报, 与 core/reward.py 同口径。

    r_t = (己方掉血, 负) + (敌方掉血, 正)

    刻意不读 ep.reward: 采集阶段那一列常常是空的（机器人只记录状态与动作）,
    而 HP 差在状态里一直都在, 随时可以重算。用状态重算还有个好处 —— 奖励
    和状态永远同源, 不会因为两处对齐不同而互相打架。
    """
    n = len(ep)
    r = np.zeros(n, dtype=np.float64)
    if n > 1:
        S = ep.state.astype(np.float32)
        r[1:] = ((S[:-1, I_HP_SELF] - S[1:, I_HP_SELF])
                 + (S[1:, I_HP_ENEMY] - S[:-1, I_HP_ENEMY]))
    return r


def _signals(ep: Episode, gamma_per_s: float, terminal_bonus: float | None):
    """算出六路信号, 每个都是长度 n 的数组。"""
    n = len(ep)
    S = ep.state.astype(np.float32)
    ts = ep.ts.astype(np.float64)

    def diff(a: np.ndarray) -> np.ndarray:
        d = np.zeros(n, dtype=np.float64)
        if n > 1:
            d[1:] = np.abs(np.diff(a.astype(np.float64)))
        return d

    # ① 血量跳变（双方合计）
    hp = diff(S[:, I_HP_SELF]) + diff(S[:, I_HP_ENEMY])

    # ② 回报落差: 事后才知道「这一步很关键」, 对应终局 MC 回报
    r = hp_reward(ep)
    if terminal_bonus:
        r[-1] += float(terminal_bonus)
    G = discounted_returns(r, ts, gamma_per_s)
    # G 是「剩余总回报」, 所以 G_t − G_{t−1} 度量的是【t−1 那一步】消耗了什么。
    # 必须把差值左移一格才对齐到事件发生的帧, 否则每个关键点都晚 50ms,
    # 而 DAgger 恰好要在那一帧接管 —— 差一格就等于教错。
    dg = np.zeros(n, dtype=np.float64)
    if n > 1:
        dg[:-1] = np.abs(np.diff(G.astype(np.float64)))
    dg = np.clip(dg / PAYOFF_SCALE_HP, 0.0, 2.0)

    # ③④ 起手 / 收招（idle 定义与 ActionHeads 一致: 两轴中位 + 无技能）
    act = ep.action.astype(np.int16)
    idle = ((act[:, 0] == AXIS_IDLE) & (act[:, 1] == AXIS_IDLE)
            & (act[:, 2] == 0))
    prev_idle = np.ones(n, dtype=bool)
    if n > 1:
        prev_idle[1:] = idle[:-1]
    engage = (~idle) & prev_idle
    disengage = idle & (~prev_idle)

    # ⑤ 威胁从无到有
    ttc = S[:, I_TTC]
    has_threat = ttc > 0.0
    prev_threat = np.zeros(n, dtype=bool)
    if n > 1:
        prev_threat[1:] = has_threat[:-1]
    threat_new = has_threat & (~prev_threat)

    # ⑥a 敌方可见性翻转
    vis = S[:, I_VISIBLE] > 0.5
    flip = np.zeros(n, dtype=bool)
    if n > 1:
        flip[1:] = vis[1:] != vis[:-1]

    # ⑥b 距离跨越交战带
    band = np.digitize(S[:, I_DIST] * DIST_FULL_PX, BAND_EDGES_PX)
    band_cross = np.zeros(n, dtype=bool)
    if n > 1:
        band_cross[1:] = band[1:] != band[:-1]

    ttc_s = np.where(has_threat, ttc * THREAT_TTC_NORM_S, 0.0)
    return {
        "hp": hp, "dg": dg, "engage": engage, "disengage": disengage,
        "threat": threat_new, "flip": flip, "band": band_cross,
        "vis": vis, "ttc_s": ttc_s, "dist_px": S[:, I_DIST] * DIST_FULL_PX,
    }


def _reasons(i: int, sig: dict, w: ImpactWeights) -> tuple[list[str], float]:
    """该帧触发了哪些信号, 按【贡献从大到小】排序 —— reasons[0] 就是主因。"""
    items: list[tuple[float, str]] = []

    if sig["hp"][i] > 1e-3:
        items.append((w.hp_swing * float(sig["hp"][i]),
                      f"hp 跳变 {sig['hp'][i]:.3f}"))
    if sig["dg"][i] > 0.02:
        items.append((w.payoff * float(sig["dg"][i]),
                      f"回报落差 {sig['dg'][i]:.2f}"))
    if sig["engage"][i]:
        items.append((w.engage, "起手"))
    if sig["disengage"][i]:
        items.append((w.disengage, "收招"))
    if sig["threat"][i]:
        items.append((w.threat, f"威胁出现 ttc={sig['ttc_s'][i]:.2f}s"))
    if sig["flip"][i]:
        items.append((w.flip, "敌方出现" if sig["vis"][i] else "敌方消失"))
    if sig["band"][i]:
        d = sig["dist_px"][i]
        where = ("近身" if d < BAND_EDGES_PX[0]
                 else ("中距" if d < BAND_EDGES_PX[1] else "远距"))
        items.append((w.distance, f"距离跨带→{where}({d:.0f}px)"))

    items.sort(key=lambda x: -x[0])
    return [t for _, t in items], float(sum(s for s, _ in items))


# ═════════════════════════ 主入口 ═════════════════════════
def extract(ep: Episode, *,
            min_gap_s: float = 0.4,
            threshold: float = 0.6,
            top_k: int | None = None,
            gamma_per_s: float = 0.98,
            terminal_bonus: float | None = None,
            weights: ImpactWeights | None = None) -> KeyframeReport:
    """提取一局的关键决策点。

    min_gap_s  相邻关键点的最小间隔（秒）。一次连段只留最强的那一帧。
    threshold  冲击力 ≤ 此值直接丢弃。默认 0.6 ≈ 单次技能命中的六成。
    top_k      只保留最强的 K 个（None = 不限）。
    terminal_bonus  终局奖励。None → 按 ep.result 自动取 win=+1 / lose=−1。
    """
    w = weights or ImpactWeights()
    n = len(ep)
    ts = ep.ts.astype(np.float64) if n else np.zeros(0)
    S = ep.state if n else np.zeros((0, 64), dtype=np.float16)

    if n == 0:
        return KeyframeReport([], 0, {}, 0.0, 0.0)

    if terminal_bonus is None:
        terminal_bonus = _RESULT_BONUS.get(ep.result, 0.0)

    sig = _signals(ep, gamma_per_s, terminal_bonus)

    impact = np.zeros(n, dtype=np.float64)
    reason_lists: list[list[str]] = [[] for _ in range(n)]
    for i in range(n):
        rs, sc = _reasons(i, sig, w)
        reason_lists[i] = rs
        impact[i] = sc

    # 时间维 NMS: 从强到弱贪心, 选中后屏蔽 ±min_gap_s 内的其它点
    order = np.argsort(-impact, kind="stable")
    blocked = np.zeros(n, dtype=bool)
    keep: list[int] = []
    for i in order:
        if impact[i] <= threshold:
            break
        if blocked[i]:
            continue
        keep.append(int(i))
        lo = int(np.searchsorted(ts, ts[i] - min_gap_s, side="left"))
        hi = int(np.searchsorted(ts, ts[i] + min_gap_s, side="right"))
        blocked[lo:hi] = True
        if top_k is not None and len(keep) >= top_k:
            break
    keep.sort()

    events = [KeyEvent(idx=i, ts=float(ts[i]), impact=round(float(impact[i]), 3),
                       reasons=tuple(reason_lists[i]), state=S[i].copy(),
                       episode_id=ep.meta.episode_id)
              for i in keep]

    by_kind: dict[str, int] = {}
    for e in events:
        kind = _primary_kind(e.reasons)
        by_kind[kind] = by_kind.get(kind, 0) + 1

    mask = eventless_mask(ep)
    return KeyframeReport(events, n, by_kind,
                          float(mask.mean()) if n else 0.0,
                          _longest_run_s(ts, mask))


def extract_many(episodes, *, per_episode_cap: int | None = None,
                 **kw) -> KeyframeReport:
    """多局合并提取。

    每局独立做 NMS（关键点不能跨局抑制）, 合并后按 impact 降序排。
    `per_episode_cap` 限制单局上限 —— 一局被打崩的比赛会产出几十个点,
    不设上限的话它会淹没其它局的全部信息。

        rep = extract_many(eps, per_episode_cap=20)
        # rep.events 里每条都带 episode_id, 能回溯到具体哪一局
    """
    events: list[KeyEvent] = []
    frames = 0
    by_kind: dict[str, int] = {}
    weighted_eventless = 0.0
    longest = 0.0
    for ep in episodes:
        rep = extract(ep, top_k=per_episode_cap, **kw)
        events.extend(rep.events)
        frames += rep.frame_count
        weighted_eventless += rep.eventless_ratio * rep.frame_count
        longest = max(longest, rep.longest_lull_s)
        for k, v in rep.by_kind.items():
            by_kind[k] = by_kind.get(k, 0) + v
    events.sort(key=lambda e: -e.impact)
    return KeyframeReport(events, frames, by_kind,
                          weighted_eventless / frames if frames else 0.0, longest)


def _primary_kind(reasons: tuple[str, ...]) -> str:
    """取第一条原因作为主分类（reasons 已按权重降序排）。"""
    if not reasons:
        return "other"
    head = reasons[0]
    for key, name in (("hp", "hp_swing"), ("回报", "payoff"), ("起手", "engage"),
                      ("收招", "disengage"), ("威胁", "threat"),
                      ("敌方", "visibility"), ("距离", "distance")):
        if head.startswith(key):
            return name
    return "other"


# ═════════════════════════ 「无事发生」的量化 ═════════════════════════
def eventless_mask(ep: Episode, *, dist_eps_px: float = 6.0) -> np.ndarray:
    """逐帧标记「无事发生」: 动作没变 + 无血量变化 + 距离几乎没动 + 无威胁。

    这就是「一局大多数都是重复操作」的量化版本。这些帧不是废物（保持时序
    连续性需要它们）, 但它们不该占据人的注意力, 也不该占满训练预算。
    """
    n = len(ep)
    if n == 0:
        return np.zeros(0, dtype=bool)
    S = ep.state.astype(np.float32)
    act = ep.action.astype(np.int16)

    same_action = np.ones(n, dtype=bool)
    if n > 1:
        same_action[1:] = np.all(act[1:] == act[:-1], axis=1)
    no_hp = np.r_[True, (np.abs(np.diff(S[:, I_HP_SELF])) < 1e-3)
                  & (np.abs(np.diff(S[:, I_HP_ENEMY])) < 1e-3)]
    dist_px = S[:, I_DIST] * DIST_FULL_PX
    still = np.ones(n, dtype=bool)
    if n > 1:
        still[1:] = np.abs(np.diff(dist_px)) < dist_eps_px
    no_threat = S[:, I_TTC] <= 0.0

    return same_action & no_hp & still & no_threat


def eventless_ratio(ep: Episode, *, dist_eps_px: float = 6.0) -> float:
    m = eventless_mask(ep, dist_eps_px=dist_eps_px)
    return float(m.mean()) if len(m) else 0.0


def _longest_run_s(ts: np.ndarray, mask: np.ndarray) -> float:
    """最长的一段连续「无事发生」持续多少秒。"""
    n = len(mask)
    if n == 0 or not mask.any():
        return 0.0
    best = 0.0
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        best = max(best, float(ts[j] - ts[i]))
        i = j + 1
    return best

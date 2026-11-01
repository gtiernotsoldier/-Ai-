# -*- coding: utf-8 -*-
"""core/episode.py — 一局对局的完整记录（离线训练的唯一事实来源）。

这个模块只有一个职责: 定义「一局数据长什么样」并把这条契约焊死。
采集器往里写, 数据集/训练脚本往外读, 三边共用同一份定义。

设计规则（每条都对应一个必然踩到的坑）:

  ① ts 是唯一时间基准
     掉帧后索引会整体错位。history 窗口必须按 ts 差值筛, 而不是按
     「往前数 k 条记录」。用索引对齐会让位置编码学到假的时序关系。

  ② valid 必须显式记录
     感知失败时状态不可信。静默沿用旧值 = 教模型忽略感知失败;
     填零 = 让模型看到不存在的局面。两者都会上线后才暴露。

  ③ 存全部帧, 不存"只变化点"
     只留变化点会让模型学到「每次决策都该换动作」, 上线后疯狂抖动。
     降权在训练侧做（sample_weights）, 不在存储侧做。

  ④ 折扣必须按时间加权
     G_t = r_t + γ^(Δt) · G_{t+1}
     按帧数累加的话, 掉帧的那几帧会被当成「瞬间过去」, γ 失去物理意义。

  ⑤ 元数据不可省
     layout_version 变了旧数据就作废; keymap_hash 变了动作标签就全错;
     git_sha 变了无法复现。这三样缺一个, 半年后都无法判断数据能不能用。

用法::

    ep = Episode(EpisodeMeta.from_env("ep_0007"))
    for ts, fid, state, valid, act, hold, r in stream:
        ep.append(ts, fid, state, valid, act, hold, r)
    ep.finish(result="win")
    ep.save("episodes/ep_0007.npz")

    ep = Episode.load("episodes/ep_0007.npz")
    print(ep.summary())
    for i in ep.change_points():          # 只在这些点上做 BC
        ...
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from core.constants import STATE_DIM
from core.policy import DURATION_BUCKETS_MS
from core.state import LAYOUT_VERSION

EPISODE_VERSION = 1

# 判定「掉帧」的阈值: 20Hz 名义间隔 50ms, 超过 1.5 倍即认为漏了一拍以上。
FRAME_GAP_S = 0.075

# 冻结后才存在的属性（access 时自动 _freeze）
_FROZEN_ATTRS = frozenset(
    {"ts", "frame_id", "state", "valid", "action", "hold_idx", "reward"})


# ═════════════════════════ 元数据 ═════════════════════════
def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_repo_root()), capture_output=True, text=True, timeout=2.0,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _keymap_hash() -> str:
    """标定文件指纹。换了 keymap 而不知情, 历史数据的动作标签会全部错位。"""
    p = _repo_root() / "config" / "keymap.json"
    try:
        return hashlib.md5(p.read_bytes()).hexdigest()[:12]
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class EpisodeMeta:
    """一局的元信息。前五项是「能否使用」的判据, 不是装饰。"""
    episode_id: str
    started_at: float                    # unix 时间戳
    duration_s: float = 0.0
    fps_nominal: float = 20.0
    layout_version: int = LAYOUT_VERSION  # 状态契约版本
    keymap_hash: str = "unknown"
    git_sha: str = "unknown"
    device: str = "unknown"
    resolution: tuple[int, int] = (1280, 720)
    result: str = "unknown"              # win / lose / timeout / abort / unknown
    note: str = ""

    @classmethod
    def from_env(cls, episode_id: str, *, device: str = "unknown",
                 resolution: tuple[int, int] = (1280, 720),
                 fps_nominal: float = 20.0) -> "EpisodeMeta":
        return cls(
            episode_id=episode_id,
            started_at=time.time(),
            fps_nominal=fps_nominal,
            layout_version=LAYOUT_VERSION,
            keymap_hash=_keymap_hash(),
            git_sha=_git_sha(),
            device=device,
            resolution=resolution,
        )

    def to_json(self) -> str:
        d = asdict(self)
        d["resolution"] = list(self.resolution)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "EpisodeMeta":
        d = json.loads(raw)
        d["resolution"] = tuple(d.get("resolution", (1280, 720)))
        return cls(**d)


# ═════════════════════════ 一局数据 ═════════════════════════
class Episode:
    """流式累积 → 一次性冻结成 numpy 数组。

    内存: 一局约 3200 帧, float16 状态 64 维 ≈ 400KB, 全量不到 1MB。
    """

    def __init__(self, meta: EpisodeMeta):
        self.meta = meta
        self.done = False
        self.result = meta.result
        self._ts: list[float] = []
        self._fid: list[int] = []
        self._state: list[np.ndarray] = []
        self._valid: list[bool] = []
        self._action: list[tuple[int, int, int, int]] = []
        self._hold: list[int] = []
        self._reward: list[float] = []
        self._frozen = False

    # ── 写入 ─────────────────────────────────────────────
    def append(self, ts: float, frame_id: int, state: np.ndarray, valid: bool,
               action: tuple[int, int, int, int] | np.ndarray,
               hold_idx: int = 3, reward: float = 0.0) -> None:
        """追加一帧。ts 必须单调递增（相对开局的秒数）。

        action = (move_ud, move_lr, skill, direction) 四头 id
        hold_idx = DURATION_BUCKETS_MS 的下标, 不是毫秒
        """
        if self._frozen:
            raise RuntimeError("已冻结, 不能再 append")
        if self._ts and ts <= self._ts[-1]:
            raise ValueError(f"ts 必须严格递增: {ts} <= {self._ts[-1]}")
        a = tuple(int(x) for x in action)
        if len(a) != 4:
            raise ValueError(f"action 需要 4 元组, 得到 {a}")
        if not 0 <= hold_idx < len(DURATION_BUCKETS_MS):
            raise ValueError(f"hold_idx 越界: {hold_idx}")
        st = np.asarray(state, dtype=np.float16).reshape(STATE_DIM)
        self._ts.append(float(ts))
        self._fid.append(int(frame_id))
        self._state.append(st)
        self._valid.append(bool(valid))
        self._action.append(a)  # type: ignore[arg-type]
        self._hold.append(int(hold_idx))
        self._reward.append(float(reward))

    def finish(self, result: str = "timeout", duration_s: float | None = None) -> None:
        self.done = True
        self.result = result
        dur = duration_s if duration_s is not None else (self._ts[-1] if self._ts else 0.0)
        self.meta = replace(self.meta, duration_s=float(dur), result=result)

    # ── 冻结 ─────────────────────────────────────────────
    def _freeze(self) -> None:
        if self._frozen:
            return
        n = len(self._ts)
        self.ts = np.asarray(self._ts, dtype=np.float32)
        self.frame_id = np.asarray(self._fid, dtype=np.int32)
        self.state = (np.stack(self._state) if n
                      else np.zeros((0, STATE_DIM), dtype=np.float16))
        self.valid = np.asarray(self._valid, dtype=bool)
        self.action = (np.asarray(self._action, dtype=np.uint8) if n
                       else np.zeros((0, 4), dtype=np.uint8))
        self.hold_idx = np.asarray(self._hold, dtype=np.uint8)
        self.reward = np.asarray(self._reward, dtype=np.float32)
        self._frozen = True

    def __len__(self) -> int:
        if self._ts:
            return len(self._ts)
        if self._frozen:                     # load 进来的: 只有数组, 没有 list
            return int(self.ts.shape[0])
        return 0

    def __getattr__(self, name: str):
        """懒冻结: append 阶段直接读 ep.state 也能工作（先冻结再返回）。"""
        if name in _FROZEN_ATTRS:
            self._freeze()
            return object.__getattribute__(self, name)
        raise AttributeError(f"{type(self).__name__!r} 没有属性 {name!r}")

    # ── 派生视图（全部按时间, 不按索引）─────────────────
    def gaps(self, threshold_s: float = FRAME_GAP_S) -> list[tuple[int, float]]:
        """返回 [(上一帧下标, 间隔秒)] —— 掉帧位置。

        这些位置是「时间断点」: history 窗口不能跨过去, 折扣要按真实 Δt 算。
        """
        if len(self) < 2:
            return []
        d = np.diff(self.ts)
        idx = np.flatnonzero(d > threshold_s)
        return [(int(i), float(d[i])) for i in idx]

    def frame_drop_rate(self, threshold_s: float = FRAME_GAP_S) -> float:
        """掉帧率: 缺失的拍数 / 应有拍数。0.05 表示丢了 5% 的帧。"""
        g = self.gaps(threshold_s)
        if not g:
            return 0.0
        nominal = 1.0 / self.meta.fps_nominal
        missing = sum(max(0, round(dt / nominal) - 1) for _, dt in g)
        expected = len(self) + missing
        return float(missing / expected) if expected else 0.0

    def change_points(self) -> np.ndarray:
        """动作发生变化的帧下标（含首帧）。BC 只在这些点上取样本。"""
        n = len(self)
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        changed = np.zeros(n, dtype=bool)
        changed[0] = True
        if n > 1:
            changed[1:] = (np.any(self.action[1:] != self.action[:-1], axis=1)
                           | (self.hold_idx[1:] != self.hold_idx[:-1]))
        return np.flatnonzero(changed)

    def window(self, t: float, span_s: float = 0.4) -> np.ndarray:
        """取 [t-span, t] 内所有有效帧的下标, 按时间升序。

        不保证等间隔 —— 调用方必须自己按 ts 处理, 这正是不能用索引的原因。
        """
        if len(self) == 0:
            return np.zeros(0, dtype=np.int64)
        mask = (self.ts <= t + 1e-9) & (self.ts > t - span_s) & self.valid
        return np.flatnonzero(mask)

    def returns(self, gamma_per_s: float = 0.98,
                terminal_bonus: float = 0.0) -> np.ndarray:
        """时间加权折扣回报 G_t = r_t + γ^(Δt) · G_{t+1}。

        γ 是「每秒」折扣, 不是每步 —— 这样掉帧不会改变折扣的物理含义。
        20Hz 下 γ=0.98/s 约等于每步 0.999。
        """
        r = self.reward.copy()
        if terminal_bonus and len(r):
            r[-1] += float(terminal_bonus)
        return discounted_returns(r, self.ts, gamma_per_s)

    def sample_weights(self, hold_weight: float = 0.3) -> np.ndarray:
        """训练采样权重: 变化点 1.0, 保持帧 hold_weight, 无效帧 0。

        注意是「降权」不是「丢弃」—— 丢掉保持帧会让模型学成每次都换动作。
        """
        n = len(self)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        w = np.full(n, float(hold_weight), dtype=np.float32)
        w[self.change_points()] = 1.0
        w[~self.valid] = 0.0
        return w

    # ── 体检 ─────────────────────────────────────────────
    def validate(self) -> list[str]:
        """返回问题清单。空列表 = 可以进训练集。"""
        self._freeze()
        errs: list[str] = []
        n = len(self)
        for name in ("ts", "frame_id", "state", "valid", "action",
                     "hold_idx", "reward"):
            arr = getattr(self, name)
            if len(arr) != n:
                errs.append(f"{name} 长度 {len(arr)} != {n}")
        if n == 0:
            errs.append("空 episode")
            return errs
        if np.any(np.diff(self.ts) <= 0):
            errs.append("ts 非严格递增")
        if np.any(np.diff(self.frame_id) < 0):
            errs.append("frame_id 回退")
        if self.state.shape[1] != STATE_DIM:
            errs.append(f"state 维度 {self.state.shape[1]} != {STATE_DIM}")
        s = self.state.astype(np.float32)
        if np.isnan(s).any():
            errs.append("state 含 NaN")
        elif np.nanmin(s) < -1e-3 or np.nanmax(s) > 1.0 + 1e-3:
            errs.append(f"state 越界 [{np.nanmin(s):.3f}, {np.nanmax(s):.3f}]")
        if self.action.shape[1] != 4:
            errs.append(f"action 列数 {self.action.shape[1]} != 4")
        if self.action[:, 3].max(initial=0) > 8:
            errs.append("direction 越界 (>8)")
        if self.hold_idx.max(initial=0) >= len(DURATION_BUCKETS_MS):
            errs.append("hold_idx 越界")
        # 时基校验: n 帧应覆盖 n/fps 秒。差超过 50% 说明录制没走节拍器,
        # 或者时钟基准搞错了 —— 这种数据算出来的折扣和时长全是假的。
        if n > 1 and self.meta.fps_nominal > 0:
            dur = float(self.ts[-1] - self.ts[0])
            expect = (n - 1) / self.meta.fps_nominal
            if expect > 0 and abs(dur / expect - 1.0) > 0.5:
                errs.append(
                    f"时基异常: {dur:.2f}s / {n} 帧, 应约 {expect:.2f}s"
                    f"（录制未走节拍器?）")
        return errs

    def summary(self) -> dict:
        self._freeze()
        cp = self.change_points()
        return {
            "episode": self.meta.episode_id,
            "frames": len(self),
            "duration_s": round(float(self.ts[-1]) if len(self) else 0.0, 2),
            "valid_ratio": round(float(self.valid.mean()) if len(self) else 0.0, 3),
            "change_points": int(len(cp)),
            "change_ratio": round(float(len(cp) / max(1, len(self))), 3),
            "frame_drop_rate": round(self.frame_drop_rate(), 4),
            "result": self.result,
            "layout_version": self.meta.layout_version,
            "keymap_hash": self.meta.keymap_hash,
            "git_sha": self.meta.git_sha,
        }

    # ── 持久化 ───────────────────────────────────────────
    def save(self, path: str | Path) -> Path:
        self._freeze()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            _episode_version=np.array(EPISODE_VERSION, dtype=np.int16),
            meta=np.array(self.meta.to_json()),
            ts=self.ts, frame_id=self.frame_id, state=self.state,
            valid=self.valid, action=self.action, hold_idx=self.hold_idx,
            reward=self.reward, done=np.array(self.done),
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Episode":
        d = np.load(str(path), allow_pickle=False)
        ver = int(d["_episode_version"])
        if ver != EPISODE_VERSION:
            raise ValueError(
                f"episode 格式版本 {ver} != 当前 {EPISODE_VERSION}, 需迁移或重录")
        meta = EpisodeMeta.from_json(str(d["meta"].item()))
        ep = cls(meta)
        ep.ts = d["ts"].astype(np.float32)
        ep.frame_id = d["frame_id"].astype(np.int32)
        ep.state = d["state"].astype(np.float16)
        ep.valid = d["valid"].astype(bool)
        ep.action = d["action"].astype(np.uint8)
        ep.hold_idx = d["hold_idx"].astype(np.uint8)
        ep.reward = d["reward"].astype(np.float32)
        ep.done = bool(d["done"])
        ep.result = meta.result
        ep._frozen = True
        return ep


# ═════════════════════════ 回报计算 ═════════════════════════
def discounted_returns(reward: np.ndarray, ts: np.ndarray,
                       gamma_per_s: float = 0.98) -> np.ndarray:
    """时间加权折扣回报 G_t = r_t + γ^(Δt) · G_{t+1}。

    模块级纯函数（不绑 Episode）: 离线分析要用它算「从状态重算的 HP 差回报」,
    而采集器写进来的 reward 列可能根本没填。两种来源必须共用同一套折扣,
    否则同一个数在两处算出不同结果, 而且很难发现。
    """
    r = np.asarray(reward, dtype=np.float64)
    n = r.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    t = np.asarray(ts, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    acc = 0.0
    for i in range(n - 1, -1, -1):
        if i < n - 1:
            acc = float(gamma_per_s) ** float(t[i + 1] - t[i]) * acc
        acc += r[i]
        out[i] = acc
    return out.astype(np.float32)


# ═════════════════════════ 批量加载 ═════════════════════════
def load_dir(path: str | Path, *, check_layout: bool = True) -> list[Episode]:
    """加载一个目录下所有 episode, 自动跳过版本/标定不符的文件。

    返回的每个 episode 都通过 validate(); 有问题的会被跳过并打印原因 ——
    静默吞掉坏数据比直接报错更危险。
    """
    out: list[Episode] = []
    for p in sorted(Path(path).glob("*.npz")):
        try:
            ep = Episode.load(p)
        except Exception as e:                       # noqa: BLE001
            print(f"[skip] {p.name}: {e}")
            continue
        if check_layout and ep.meta.layout_version != LAYOUT_VERSION:
            print(f"[skip] {p.name}: layout_version "
                  f"{ep.meta.layout_version} != {LAYOUT_VERSION}")
            continue
        errs = ep.validate()
        if errs:
            print(f"[skip] {p.name}: {errs}")
            continue
        out.append(ep)
    return out

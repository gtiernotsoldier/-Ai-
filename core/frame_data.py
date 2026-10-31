# -*- coding: utf-8 -*-
"""M4.6 — 帧数据表 (frame_data)

打磨清单⑥的地基: 每个招式 startup/active/recovery/cancel_from/cancel_into。
  - 取消后摇: 下一招在上一招 cancel_from 帧之后即可接入, 不必等 recovery 打满
  - 连段衔接: Sequencer 用 gap_to() 把帧表换算成步骤间的精确毫秒间隔
  - 卡替身(⑦): punish 连段该在对方硬直的第几帧插入, 也由本表推算

全部数值为占位标定值(见 config/frame_data.json _meta.note), 帧号以按下=0 计。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FRAME_DATA_PATH = (Path(__file__).resolve().parent.parent
                           / "config" / "frame_data.json")


class FrameDataError(ValueError):
    pass


@dataclass(frozen=True)
class FrameMove:
    """一个招式的帧数据。帧号从按下瞬间=0 起计。"""
    name: str
    startup: int                       # 前摇帧: 动作已起、判定未生效
    active: int                        # 判定生效帧
    recovery: int                      # 后摇帧: 判定已过、不能动
    cancel_from: int                   # 取消窗口起点帧(startup ≤ 此值 ≤ total)
    cancel_into: tuple[str, ...]       # 可取消后接入的招式

    @property
    def total(self) -> int:
        return self.startup + self.active + self.recovery

    def phase(self, frame: int) -> str:
        if frame < self.startup:
            return "startup"
        if frame < self.startup + self.active:
            return "active"
        return "recovery"

    def gap_to(self, nxt: "FrameMove") -> int:
        """两招衔接的帧间隔: 下一招可取消进 → cancel_from(取消后摇), 否则打满 total。"""
        return self.cancel_from if nxt.name in self.cancel_into else self.total


class FrameData:
    """帧数据表: 加载 + 查询 + 一致性校验(坏表早炸, 不留到运行期)。

    构造函数直接收 raw dict —— 测试可注入小帧表(短 gap, 秒级真跑)。
    """

    def __init__(self, raw: dict):
        meta = raw.get("_meta", {})
        # fps 允许写在 _meta(生产表)或顶层(测试注入的 mini 表)
        self.fps = int(meta.get("fps", raw.get("fps", 60)))
        self.frame_ms = 1000.0 / self.fps
        self.frame_s = self.frame_ms / 1000.0    # 帧长(秒): clock 时间戳口径
        self._moves: dict[str, FrameMove] = {}
        for name, m in raw.get("actions", {}).items():
            mv = FrameMove(
                name=name,
                startup=int(m["startup"]), active=int(m["active"]),
                recovery=int(m["recovery"]), cancel_from=int(m["cancel_from"]),
                cancel_into=tuple(m.get("cancel_into", ())),
            )
            if not (0 <= mv.startup <= mv.cancel_from <= mv.total):
                raise FrameDataError(
                    f"{name}: 帧约束 startup≤cancel_from≤total 不成立 "
                    f"({mv.startup}/{mv.cancel_from}/{mv.total})")
            self._moves[name] = mv
        self._combos: dict[str, tuple[str, ...]] = {}
        for name, c in raw.get("combos", {}).items():
            self._combos[name] = tuple(c["steps"])
        self._validate_refs()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "FrameData":
        p = Path(path) if path else DEFAULT_FRAME_DATA_PATH
        if not p.exists():
            raise FrameDataError(f"帧数据表不存在: {p}")
        with open(p, encoding="utf-8") as f:
            return cls(json.load(f))

    def _validate_refs(self) -> None:
        for name, mv in self._moves.items():
            for nxt in mv.cancel_into:
                if nxt not in self._moves:
                    raise FrameDataError(
                        f"{name}: cancel_into 引用了不存在的招 {nxt!r}")
        for name, steps in self._combos.items():
            for b in steps:
                if b not in self._moves:
                    raise FrameDataError(
                        f"连段 {name}: 引用了不存在的招 {b!r}")

    # ── 查询 ──
    def move(self, name: str) -> FrameMove:
        try:
            return self._moves[name]
        except KeyError:
            raise FrameDataError(
                f"未知招式 {name!r}（已有: {sorted(self._moves)}）") from None

    def has(self, name: str) -> bool:
        return name in self._moves

    @property
    def move_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._moves))

    def combo(self, name: str) -> tuple[str, ...]:
        try:
            return self._combos[name]
        except KeyError:
            raise FrameDataError(
                f"未知连段 {name!r}（已有: {sorted(self._combos)}）") from None

    @property
    def combo_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._combos))

    def gap_s(self, cur: str, nxt: str) -> float:
        """两招衔接间隔(秒) = 帧间隔 × 帧长。"""
        return self.move(cur).gap_to(self.move(nxt)) * self.frame_ms / 1000.0

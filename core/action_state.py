# -*- coding: utf-8 -*-
"""M4.6 — 动作时序状态机 (action_state)

打磨清单⑥第 2 项: 按下技能后要能回答「现在是这个动作的第几帧、
取消窗口开了没」。没有它, 取消后摇只能瞎猜时机。

时间基准 = Sequencer 编译序列时的**计划按下时刻**。序列型宏
humanize=False 零反应延迟、due 预排程确定; 1% 失误扰动只偏坐标不偏
时刻 → 计划时刻 ≈ 实际时刻(线程启动偏差 ~0.1ms, 可忽略)。

状态流转(单招): IDLE → STARTUP → ACTIVE → RECOVERY → IDLE
取消衔接: 下一招的按下时刻 = 上一招 cancel_from 帧 → 上一招 RECOVERY
段被截断, query 自动切到最新按下的招(后取消前的语义)。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.frame_data import FrameData


class ActionPhase(Enum):
    IDLE = "idle"
    STARTUP = "startup"
    ACTIVE = "active"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class PhaseInfo:
    """某时刻的动作阶段快照。"""
    button: str | None          # 当前在飞的招; IDLE 时 None
    phase: ActionPhase
    frame: int                  # 招内帧号(按下=0); IDLE 时为已过招的越界帧
    cancel_open: bool           # 取消窗口是否已开(该招 cancel_from ≤ frame < total)
    pending: bool               # 有已排程但尚未按下的后续步骤(两招衔接等待中)


_IDLE = PhaseInfo(button=None, phase=ActionPhase.IDLE, frame=0,
                  cancel_open=False, pending=False)


class ActionTracker:
    """按下时刻起表的动作阶段机。由 Sequencer 在提交序列时写入计划表。"""

    def __init__(self, fd: FrameData):
        self.fd = fd
        self._plan: tuple[tuple[float, str], ...] = ()   # (press_ts, button) 时刻升序

    # ── 计划写入/清除 ──
    def begin(self, plan: tuple[tuple[float, str], ...]) -> None:
        """写入整条序列的计划按下时刻(绝对时钟, 升序, 不可为空)。"""
        if not plan:
            raise ValueError("计划表不可为空")
        self._plan = tuple(plan)

    def clear(self) -> None:
        self._plan = ()

    @property
    def has_plan(self) -> bool:
        return bool(self._plan)

    # ── 查询 ──
    def query(self, now: float) -> PhaseInfo:
        """now 时刻的动作阶段。"""
        if not self._plan:
            return _IDLE
        # 最后一个已到按下时刻的步骤 = 当前在飞的招(后按下取消先按下)
        idx = -1
        for i, (ts, _b) in enumerate(self._plan):
            if ts <= now:
                idx = i
            else:
                break
        if idx < 0:
            # 第一步还没到按下时刻(宏观上极少见: 提交即排程 t0)
            return PhaseInfo(None, ActionPhase.IDLE, 0, False, pending=True)
        ts, button = self._plan[idx]
        mv = self.fd.move(button)
        # 1µs 容差吸收浮点误差(如 100.020-100.0 的 double 差落在 1.99 帧);
        # 对 10~16.7ms 帧长无影响
        frame = int((now - ts) / self.fd.frame_s + 1e-6)   # now 口径=秒
        if frame >= mv.total:
            # 本招打完(未被下一招取消): 下一招在等待 gap 中则 pending
            pending = idx + 1 < len(self._plan)
            return PhaseInfo(None, ActionPhase.IDLE, frame, False, pending=pending)
        cancel_open = frame >= mv.cancel_from
        ph = (ActionPhase.STARTUP if frame < mv.startup
              else (ActionPhase.ACTIVE if frame < mv.startup + mv.active
                    else ActionPhase.RECOVERY))
        return PhaseInfo(button, ph, frame, cancel_open, pending=False)

    def cancel_window_open(self, now: float, into: str) -> bool:
        """now 时刻是否处于「可被 into 取消」的窗口内(⑥ 取消后摇判定)。"""
        q = self.query(now)
        if q.button is None or not q.cancel_open:
            return False
        return into in self.fd.move(q.button).cancel_into

    def busy(self, now: float) -> bool:
        """有招在飞, 或下一步已在排程等待中(此时插入新序列必须先打断)。"""
        q = self.query(now)
        return q.button is not None or q.pending

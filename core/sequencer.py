# -*- coding: utf-8 -*-
"""M4.6 — 帧级执行层 (sequencer)

打磨清单第五节的架构落地: 20Hz 策略层出**意图**, 本层按**帧表**精确执行。

旧用法的三个浪费, 在这里逐一解决:
  1. 连段 = 一次编译 + 一次下发(Macro humanize=False 序列型, due 预排程),
     中途只在需要分支/取消时才打断 —— MacroEngine 的精确排程能力真正用起来
     (旧: Orchestrator 每 tick 重建宏, 精确排程完全被丢掉)
  2. ActionTracker 回答「第几帧/取消窗口开了没」→ 新意图只在取消窗口内
     替换(取消后摇), 其余时刻忽略(防 20Hz 刷屏把连段拆散)
  3. 事件分支(⑦ 卡替身): sub_flash → punish 不经过 20Hz 策略回合,
     on_events() 直接打断剩余步骤切分支序列 —— 帧级响应

走位不归本层管: JoystickStream 持久流由 Orchestrator 持有, 方向意图每
tick 在线更新; 本层只负责按钮序列(CONTACT_BUTTON 路), 与摇杆并行不互顶。
"""
from __future__ import annotations

from dataclasses import dataclass

from core.action_state import ActionTracker
from core.executor import Executor, MacroHandle
from core.frame_data import FrameData
from core.macro import Macro, MacroEngine, MonotonicClock, VirtualClock, Tap, Wait
from core.constants import CONTACT_BUTTON

# 同触点步间最小保护间隔: 必须保证上一步的抬起(u)先于下一步的按下(d),
# 否则同一路触点上 d 会顶掉还没发出的 u。
MIN_STEP_GAP_S = 0.02


@dataclass(frozen=True)
class Step:
    """序列中的一步: 按一个按钮。"""
    button: str
    hold_ms: float = 40.0


@dataclass(frozen=True)
class Sequence:
    """预排程序列(连段/意图): 一次编译一次下发。

    branches: (事件名, 替代连段名) —— ⑦ 两阶段序列的事件分支。
      例: bait 序列声明 ("enemy_sub", "punish_sub"), 运行中看到敌方替身闪
      → 打断剩余 bait 步骤, 立刻切 punish 连段(帧级响应, 不走 20Hz 回合)。
    """
    name: str
    steps: tuple[Step, ...]
    branches: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError(f"序列 {self.name!r} 至少需要一步")

    @property
    def first_button(self) -> str:
        return self.steps[0].button


def combo_sequence(fd: FrameData, name: str,
                   branches: tuple[tuple[str, str], ...] = ()) -> Sequence:
    """从帧数据表的连段定义构造 Sequence(步间 gap 由帧表推算, 见 _compile)。"""
    return Sequence(name, tuple(Step(b) for b in fd.combo(name)), tuple(branches))


class Sequencer:
    """60Hz 帧级执行层: 接意图 → 编译成宏 → 一次下发 → 按帧表精确触发。"""

    JOIN_TIMEOUT_S = 0.05          # 打断旧宏后的 join 上限(中断协议即时唤醒)

    def __init__(self, executor: Executor, fd: FrameData,
                 clock: MonotonicClock | VirtualClock | None = None):
        self.executor = executor
        self.fd = fd
        self.tracker = ActionTracker(fd)
        self._clock = clock if clock is not None else MonotonicClock()
        self._handle: MacroHandle | None = None
        self._seq: Sequence | None = None
        # 健康度统计(进 LoopReport / 测试断言)
        self.stats = {"submitted": 0, "rejected_busy": 0,
                      "cancel_swap": 0, "branch_taken": 0}

    def _handle_alive(self) -> bool:
        return self._handle is not None and not self._handle.done

    # ── 意图入口(20Hz 策略层调用) ──
    def submit(self, seq: Sequence, force: bool = False) -> bool:
        """提交序列。返回是否被接受。

        空闲        → 直接跑。
        在飞+force  → 立即打断(逃命级/事件分支级, 不看窗口)。
        在飞+取消窗口开 → 打断剩余, 换新序列(**取消后摇**)。
        在飞+窗口未开 → 拒绝(rejected_busy), 当前连段继续跑完
                        —— 20Hz 刷屏不会把连段拆散。

        「在飞」的判定以 ActionTracker(帧表语义)为准, 而非宏线程是否结束:
        单步 Tap 宏 ~40ms 就发完 down/up, 但这一招在游戏里要打到 total 帧
        —— 后摇段正是「取消后摇」的窗口, 必须继续视为在飞。
        """
        now = self._clock.now()
        busy = self._handle_alive() or self.tracker.busy(now)
        if busy:
            if force:
                pass                                   # 分支/紧急: 无条件打断
            elif self.tracker.cancel_window_open(now, seq.first_button):
                self.stats["cancel_swap"] += 1         # 取消后摇替换
            else:
                self.stats["rejected_busy"] += 1
                return False
            if self._handle is not None:           # handle 可能已结束(帧表仍在飞)
                self._handle.interrupt()
                self._handle.join(timeout=self.JOIN_TIMEOUT_S)
        self._seq = seq
        self._handle = self.executor.play_async(self._compile(seq))
        self.tracker.begin(self._plan_presses(seq, now))
        self.stats["submitted"] += 1
        return True

    # ── 帧级事件分支(⑦ 卡替身) ──
    def on_events(self, events) -> bool:
        """感知层的事件边沿(如 enemy_sub) → 切当前序列声明的分支。

        从「看到替身闪」到「punish 首步按下」不经过任何 20Hz 回合 ——
        bait 打断 + punish 提交在同一次调用内完成。
        在飞判定同 submit: 帧表语义(后摇段也响应分支, 等于用 punish
        取消后摇)。
        """
        if self._seq is None:
            return False
        if not (self._handle_alive() or self.tracker.busy(self._clock.now())):
            return False
        for ev in events:
            for ev_name, combo in self._seq.branches:
                if ev == ev_name:
                    if self._handle is not None:
                        self._handle.interrupt()
                        self._handle.join(timeout=self.JOIN_TIMEOUT_S)
                    self.stats["branch_taken"] += 1
                    self._seq = None
                    self._handle = None
                    self.tracker.clear()           # 旧计划作废, 防误判 busy
                    return self.submit(combo_sequence(self.fd, combo), force=True)
        return False

    # ── 编译: 帧表 → 宏 ──
    def _step_gap_s(self, cur: Step, nxt: Step) -> float:
        """步间间隔: 帧表 gap(取消衔接取 cancel_from, 否则打满 total),
        且不小于同触点保护间隔(上一步 u 必须先于下一步 d)。"""
        return max(self.fd.gap_s(cur.button, nxt.button),
                   cur.hold_ms / 1000.0 + MIN_STEP_GAP_S)

    def _compile(self, seq: Sequence) -> Macro:
        """把序列编译成一个序列型宏: Tap(步骤) + Wait(步进 gap) 交错。

        ⚠️ 排程语义: MacroEngine 的 offset 只被 Wait 推进, Tap 的事件
        (0=d, hold=u)相对**当前 offset** 排布 —— 所以 Wait 必须放
        **步进 gap**(相对上一步的间隔), 不是累计时刻, 否则间隔二次累加。
        humanize=False —— 连段内部事件零拟人延迟(宪法「拟人化分层」:
        反应延迟只属于反应型动作, 连段是肌肉记忆一次排好的)。
        1% 指令级失误扰动仍然生效(只偏坐标, 不偏时刻)。
        """
        prims: list[Tap | Wait] = []
        gap = 0.0
        for i, st in enumerate(seq.steps):
            if i > 0:
                prims.append(Wait(gap * 1000.0))   # Wait 单位=毫秒(macro.py 契约)
            cx, cy, _r = self.executor.km.circle(f"btn_{st.button}")
            prims.append(Tap(cx, cy, contact=CONTACT_BUTTON, hold_ms=st.hold_ms))
            if i + 1 < len(seq.steps):
                gap = self._step_gap_s(st, seq.steps[i + 1])
        return Macro(f"seq_{seq.name}", tuple(prims), humanize=False)

    def _plan_presses(self, seq: Sequence,
                      t0: float) -> tuple[tuple[float, str], ...]:
        """计划按下时刻表(绝对时钟): 与 _compile 的 Wait 排程严格一致。

        t0 由 submit 传入(与宏线程的 clock.now() 差 ~线程启动 0.1ms, 可忽略),
        保持 tracker 计划与宏排程同基准。
        """
        presses: list[tuple[float, str]] = []
        t = 0.0
        for i, st in enumerate(seq.steps):
            presses.append((t0 + t, st.button))
            if i + 1 < len(seq.steps):
                t += self._step_gap_s(st, seq.steps[i + 1])
        return tuple(presses)

    # ── 状态查询(策略层/报告用) ──
    @property
    def busy(self) -> bool:
        return (self._handle is not None and not self._handle.done
                and self.tracker.busy(self._clock.now()))

    @property
    def current_name(self) -> str | None:
        return self._seq.name if self._seq is not None else None

    # ── 收尾 ──
    def shutdown(self) -> None:
        """打断在飞序列(中断协议: 只释放自己的触点, 零残留)。"""
        if self._handle is not None:
            self._handle.interrupt()
            self._handle.join(timeout=1.0)
            self._handle = None
        self._seq = None
        self.tracker.clear()

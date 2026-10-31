# -*- coding: utf-8 -*-
"""M4.5 — 20Hz 闭环主循环 (orchestrator)

节拍（宪法 DECISION_HZ=20）:
  捕获 → 感知(快60Hz/慢12Hz) → 状态组装 → 决策(看门狗20ms沿用上帧) → 下发执行

下发分两路（P0-1 修复后的形态）:
  走位 → JoystickStream 持久流: 整局只按下一次, 之后每 tick 在线更新方向,
         转向不释放触点、不重建宏。方向变化时可提前下发（下限 20ms）。
  按钮 → play_async 一次性宏, 走独立触点(CONTACT_BUTTON): 与摇杆并行不互顶,
         换按钮时才走中断协议（M4 中断零残留）。

闭环纪律: 主循环内零网络/零磁盘/零网络注入; pacing sleep 仅是实时节拍器
（等价 smoothness_test 的离线 sleep 口径）。

python -m core.orchestrator  →  沙箱全链路自走演示 + 闭环延迟报告
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from core.constants import (
    CHARGE_HOLD_MS, DECISION_PERIOD_MS, DECISION_TIMEOUT_MS, LOOP_BUDGET_MS,
    REDLINE_LATENCY_MS, STEER_EPS, TARGET_LATENCY_MS,
)
from core.executor import Executor, MacroHandle
from core.injection import MockInjector, TouchInjector
from core.keymap import Keymap
from core.macro import JoystickStream, MonotonicClock, VirtualClock
from core.perception import Perception
from core.policy import Action, PolicyProvider
from core.screen_capture import FrameSource, SyntheticSource
from core.sequencer import Sequencer, Sequence, Step
from core.state import StateBuilder
from core.strategy import IntentPolicy
from core.synthetic import Scene


# ═════════════════════════ 闭环报告 ═════════════════════════
@dataclass
class LoopReport:
    ticks: int = 0
    actual_hz: float = 0.0
    watchdog_hits: int = 0
    action_counts: dict = field(default_factory=dict)
    # 摇杆持久流健康度（P0-1 回归哨兵）:
    #   stick_starts 连续走位期间应为 1（按下一次, 之后只转向）。
    #   若它接近 ticks, 说明摇杆在被反复重建 —— 旧 bug 的形态。
    stick_starts: int = 0
    stick_events: int = 0
    stick_steers: int = 0
    # 各阶段耗时(ms)样本
    capture_ms: list = field(default_factory=list)
    perceive_ms: list = field(default_factory=list)
    state_ms: list = field(default_factory=list)
    decide_ms: list = field(default_factory=list)
    issue_ms: list = field(default_factory=list)
    loop_ms: list = field(default_factory=list)     # 帧就绪(frame.ts) → 下发完成
    full_ms: list = field(default_factory=list)     # 捕获起点 → 下发完成
    # M4.6 Sequencer 健康度: submitted/rejected_busy/cancel_swap/branch_taken
    seq_stats: dict = field(default_factory=dict)

    def _p(self, xs: list, q: float) -> float:
        return float(np.percentile(xs, q)) if xs else 0.0

    def render(self) -> str:
        rows = [
            ("画面捕获", self.capture_ms, 10.0),
            ("感知推理", self.perceive_ms, 15.0),
            ("状态组装", self.state_ms, 1.0),
            ("决策(看门狗20ms)", self.decide_ms, 15.0),
            ("动作下发", self.issue_ms, 1.0),
            ("感知→下发(帧就绪起)", self.loop_ms, 32.0),
            ("闭环合计(捕获起)", self.full_ms, LOOP_BUDGET_MS),
        ]
        lines = [f"ticks={self.ticks}  实测节拍={self.actual_hz:.2f}Hz  "
                 f"看门狗触发={self.watchdog_hits}次",
                 "动作分布: " + (", ".join(f"{k}×{v}" for k, v in
                                     sorted(self.action_counts.items())) or "-"),
                 f"摇杆持久流: 按下{self.stick_starts}次  触点事件{self.stick_events}个"
                 f"（其中提前转向{self.stick_steers}个）  "
                 f"{'PASS' if self.stick_starts <= 1 else 'FAIL(反复重建)'}",
                 *( ["序列器: " + ", ".join(f"{k}={v}" for k, v in
                                           sorted(self.seq_stats.items()))]
                    if self.seq_stats else [] ),
                 f"{'环节':<22}{'预算':>7}{'p50':>9}{'p95':>9}{'max':>9}  判定"]
        ok_all = True
        for name, xs, budget in rows:
            p50, p95, mx = self._p(xs, 50), self._p(xs, 95), max(xs) if xs else 0.0
            ok = p95 <= budget
            ok_all &= ok
            lines.append(f"{name:<22}{budget:>7.1f}{p50:>9.2f}{p95:>9.2f}"
                         f"{mx:>9.2f}  {'PASS' if ok else 'FAIL'}")
        pct_red = self._p(self.full_ms, 95) / REDLINE_LATENCY_MS * 100
        pct_tgt = self._p(self.full_ms, 95) / TARGET_LATENCY_MS * 100
        lines.append(f"闭环 p95 占红线 {pct_red:.1f}% / 目标 {pct_tgt:.1f}% "
                     f"(不含模拟器渲染16ms+响应16ms 物理税)")
        return "\n".join(lines)


# ═════════════════════════ 主循环 ═════════════════════════
class Orchestrator:
    """20Hz 感知→决策→执行闭环。"""

    def __init__(self, source: FrameSource, perception: Perception,
                 builder: StateBuilder, policy: PolicyProvider,
                 executor: Executor,
                 clock: MonotonicClock | VirtualClock | None = None,
                 sequencer: Sequencer | None = None):
        self.source = source
        self.perception = perception
        self.builder = builder
        self.policy = policy
        self.executor = executor
        self.clock = clock if clock is not None else MonotonicClock()
        self.sequencer = sequencer
        self._stop = threading.Event()
        # 走位: 长生命周期摇杆持久流（按下一次, 之后在线转向, 绝不因转向而释放）
        self._stream: JoystickStream | None = None
        # 按钮: 一次性宏, 走独立触点(CONTACT_BUTTON), 与摇杆并行不互顶
        self._btn_handle: MacroHandle | None = None
        self._btn_name: str | None = None
        self._stick_starts: int = 0        # 摇杆被(重新)按下次数, 写入 LoopReport
        self._last_watchdog: bool = False  # 上帧决策看门狗是否超时 → state.meta.watchdog
        # M4.6: 策略层若声明 bind_sequencer(意图式策略), 把执行层交给它 ——
        # 策略可 submit 连段/bait 序列, 而不只单帧按钮（清单第五节分层）
        if sequencer is not None and hasattr(policy, "bind_sequencer"):
            policy.bind_sequencer(sequencer)

    # ── 按钮反馈（打磨清单⑤: state.buttons 切片的真值来源）──
    def _buttons_feedback(self, action: Action | None) -> dict | None:
        """上一帧实际下发的动作 → buttons 反馈 dict（供 StateBuilder）。

        策略层由此知道"我上一帧按了什么、摇杆在哪、蓄力蓄到哪",
        外加 action_progress(当前在飞招式帧进度)与 watchdog(上帧前向是否超时)。
        """
        if action is None:
            return None
        return {
            "joy_x": action.move[0], "joy_y": action.move[1],
            "atk": action.button == "atk",
            "skill": action.button in ("skill1", "skill2", "skill3", "ult"),
            "dash": action.button == "dash",
            "charged": (action.button is not None
                        and action.hold_ms >= CHARGE_HOLD_MS),
            "action_progress": self._action_progress(),
            "watchdog": self._last_watchdog,
        }

    def _action_progress(self) -> float:
        """当前在飞招式的帧进度 0~1（来自 Sequencer 的 ActionTracker）。"""
        if self.sequencer is None:
            return 0.0
        return self.sequencer.progress()

    def _issue(self, action: Action) -> float:
        """走位与按钮分路下发。

        走位 → 摇杆持久流: 首次按下一次, 之后每 tick 只是在线更新方向。
                不中断、不重建、不释放触点。这是 P0-1 的修复点。
        按钮 → 独立触点异步宏: 同一按钮在飞时不重复触发; 换按钮时才打断旧的。
                摇杆不受影响 → 「边走边放技能」成立。

        旧实现用 `self._last_action == action`（浮点元组相等）判同向续流,
        方向差一个尾数就中断重建摇杆宏; 而重建的宏首事件前有 120ms 拟人延迟
        > 50ms 节拍 → 每个宏都在发出按下事件前就被打断。实测方向每 tick 变
        0.05° 时, 2 秒内按下事件数为 0（智能体完全不动）。详见 JoystickStream。
        """
        t0 = time.perf_counter()
        dx, dy = action.move

        # ── 走位: 在线转向 ──
        if abs(dx) > STEER_EPS or abs(dy) > STEER_EPS:
            if self._stream is None:
                self._stream = self.executor.joystick_stream()
            if not self._stream.alive:
                self._stream.start()          # 仅此处有一次 120ms 反应延迟
                self._stick_starts += 1
            self._stream.set_direction(dx, dy, action.strength)
        elif self._stream is not None and self._stream.alive:
            self._stream.set_direction(0.0, 0.0)   # 回中: 停走但手指仍握杆

        # ── 按钮: 按钮意图两路 ──
        if action.button is not None:
            if self.sequencer is not None:
                # M4.6 分层: 按钮意图交给 Sequencer —— 空闲即发, 取消窗口内
                # 替换(取消后摇), 窗口未开则忽略(不逐 tick 重建/拆散连段)
                ok = self.sequencer.submit(
                    Sequence("tick", (Step(action.button,
                                           hold_ms=action.hold_ms),)))
                if ok:
                    # 自己按了替身/闪避 → 自侧事件计时器起表（打磨清单①⑧:
                    # 真机这两者无 HUD 区域, 只能按过什么起什么表）
                    if action.button == "sub":
                        self.perception.trigger_event("self_sub")
                    elif action.button == "dash":
                        self.perception.trigger_event("self_dash")
            else:
                in_flight = (self._btn_handle is not None
                             and not self._btn_handle.done)
                if not (in_flight and self._btn_name == action.button):
                    if in_flight:                 # 换按钮 → 打断旧的（取消语义）
                        self._btn_handle.interrupt()
                        self._btn_handle.join(timeout=0.05)
                    self._btn_handle = self.executor.play_async(
                        self.executor.tap_macro(action.button,
                                                hold_ms=action.hold_ms))
                    # 自己按了替身 → 自侧替身计时器起表（打磨清单①: 知道自己按过什么）
                    if action.button == "sub":
                        self.perception.trigger_event("self_sub")
                    elif action.button == "dash":
                        self.perception.trigger_event("self_dash")
                self._btn_name = action.button
        return (time.perf_counter() - t0) * 1000.0

    def stop(self) -> None:
        self._stop.set()

    def run(self, ticks: int | None = None,
            duration_s: float | None = None) -> LoopReport:
        """跑闭环: 指定 tick 数或时长(秒), 二者均缺省=跑 1 秒。"""
        rpt = LoopReport()
        period = DECISION_PERIOD_MS / 1000.0
        t_start = self.clock.now()
        t_deadline = (t_start + duration_s) if duration_s else None
        last_action: Action | None = None
        last_watchdog = False
        n = 0
        try:
            with self.source:
                while not self._stop.is_set():
                    if ticks is not None and n >= ticks:
                        break
                    if t_deadline is not None and self.clock.now() >= t_deadline:
                        break
                    t_tick = self.clock.now()

                    # ── 感知链路 ──
                    t0 = time.perf_counter()
                    frame = self.source.read()
                    rpt.capture_ms.append((time.perf_counter() - t0) * 1000)
                    t_capture0 = t0                     # 捕获起点(含渲染)
                    t0 = time.perf_counter()
                    res = self.perception.update(frame)
                    rpt.perceive_ms.append((time.perf_counter() - t0) * 1000)
                    # M4.6: 事件边沿 → 帧级分支(⑦ 卡替身): 在决策回合之外直接
                    # 切 punish —— 从看到替身闪到 punish 首步不经 20Hz 等待
                    if self.sequencer is not None and res.events_fired:
                        self.sequencer.on_events(res.events_fired)
                    t0 = time.perf_counter()
                    state = self.builder.build(res, frame.frame_id, frame.ts,
                                               buttons=self._buttons_feedback(last_action))
                    rpt.state_ms.append((time.perf_counter() - t0) * 1000)

                    # ── 决策 + 看门狗: 超时沿用上帧动作 ──
                    t0 = time.perf_counter()
                    action = self.policy.decide(state)
                    decide_ms = (time.perf_counter() - t0) * 1000
                    rpt.decide_ms.append(decide_ms)
                    watchdog_hit = decide_ms > DECISION_TIMEOUT_MS
                    if watchdog_hit:
                        rpt.watchdog_hits += 1
                        action = last_action if last_action is not None else action
                    self._last_watchdog = watchdog_hit

                    # ── 下发 ──
                    rpt.issue_ms.append(self._issue(action))
                    key = action.button if action.button is not None else "move"
                    rpt.action_counts[key] = rpt.action_counts.get(key, 0) + 1
                    last_action = action
                    # loop_ms 口径: 帧就绪(frame.ts, monotonic) → 下发完成。
                    # 必须与 frame.ts 同一时钟基准 —— 混用 perf_counter 时
                    # Linux 下二者恰好同基准(实测差 ~8e-7s), 跨平台(Windows QPC)
                    # 这个数字会整体失真。duration 类指标是差值, 不受基准影响。
                    _now_mono = time.monotonic()
                    _now = time.perf_counter()
                    rpt.loop_ms.append((_now_mono - frame.ts) * 1000)
                    rpt.full_ms.append((_now - t_capture0) * 1000)
                    n += 1
                    rpt.ticks = n

                    # ── 实时节拍器 ──
                    remain = period - (self.clock.now() - t_tick)
                    if remain > 0:
                        self.clock.wait(remain, self._stop)
        finally:
            # 收尾: 停摇杆持久流(精确释放 CONTACT_JOYSTICK) + 中断在飞按钮宏,
            # 两路都清干净才算零残留。
            if self._stream is not None:
                self._stream.stop(timeout=1.0)
                rpt.stick_events = self._stream.n_events
                rpt.stick_steers = self._stream.n_steers
            rpt.stick_starts = self._stick_starts
            if self.sequencer is not None:
                self.sequencer.shutdown()            # 打断在飞序列(只释放自己的触点)
                rpt.seq_stats = dict(self.sequencer.stats)
            if self._btn_handle is not None:
                self._btn_handle.interrupt()
                self._btn_handle.join(timeout=1.0)
            self.executor.release_all()          # 幂等兜底
        elapsed = self.clock.now() - t_start
        rpt.actual_hz = rpt.ticks / elapsed if elapsed > 0 else 0.0
        return rpt


# ═════════════════════════ 沙箱自走演示 ═════════════════════════
def main(duration_s: float = 3.0) -> int:
    print("=" * 78)
    print("Shinobi_God 闭环演示 (M1-M4.7 全链路: 捕获→感知→状态→博弈决策→Sequencer→执行)")
    print("=" * 78)
    src = SyntheticSource(fps=60, scene=Scene())
    from core.detectors import MockFastDetector
    from core.frame_data import FrameData
    from core.perception import CVReader
    perc = Perception(MockFastDetector(), CVReader(Keymap.load()), slow_every=5)
    builder = StateBuilder()
    # ⑨ 博弈策略: 躲避/逃墙角/逼墙/bait-punish/trade 评估（RulePolicy 保留作基线）
    policy = IntentPolicy(fd=FrameData.load())
    inj = MockInjector(clock=time.monotonic)
    executor = Executor(inj, Keymap.load(), humanize=True)
    seqr = Sequencer(executor, FrameData.load())
    orch = Orchestrator(src, perc, builder, policy, executor, sequencer=seqr)

    rpt = orch.run(duration_s=duration_s)
    print()
    print(rpt.render())
    residual = inj.contacts_pressed()
    print(f"残留触点: {len(residual)} → {'PASS' if not residual else 'FAIL'}")
    # stick_starts<=1: 整段走位期间摇杆只按下一次（P0-1 回归哨兵）。
    # 旧实现下这个数会等于 tick 数级别, 且触点事件数为 0（智能体完全不动）。
    ok = (rpt.ticks >= duration_s * 15 and not residual
          and rpt._p(rpt.full_ms, 95) <= LOOP_BUDGET_MS
          and rpt.stick_starts <= 1)
    print("\n纪律提醒: 闭环内零网络/零磁盘; pacing sleep 仅是实时节拍器。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""M4.5 — 决策层接口 (policy)

契约: PolicyProvider.decide(state: float32(64,)) -> Action
实现:
  RulePolicy  规则策略（读 64 维状态切片, 沙箱/联调/基线用）
  OnnxPolicy  student.onnx FP16 前向（真推理, 缺 onnxruntime 优雅报错）
  SlowPolicy  测试替身（注入延迟 → 看门狗验证）

看门狗（宪法）: 前向超时 20ms → 沿用上帧动作。超时判定在 Orchestrator 完成。
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from core.constants import ACTION_BUTTONS
from core.exceptions import DependencyMissing
from core.state import DEFAULT_LAYOUT, StateLayout


# ═════════════════════════ 动作空间 ═════════════════════════
@dataclass(frozen=True)
class Action:
    """单帧动作: 摇杆方向 + 可选按钮。二者可同时存在（边走边放技能）。

    move      方向单位向量, (0,0)=回中停走
    strength  推杆幅度 [0,1]。满推=冲刺走位, 半推=慢速压场/微调。
              方向决定"往哪走", strength 决定"走多急" —— 压场靠的是后者。
    hold_ms   按钮按压时长。点按 ≈40ms; 长按/蓄力 ≥ CHARGE_HOLD_MS。
              普攻的长按与点按在-game 是两套行为, 必须能表达。
    duration_ms 摇杆流刷新窗口(≈4 决策周期), 保留字段供离散动作表使用。

    走位由 JoystickStream 持久流承载: 同向/微变向都是在线更新, 不重建不释放;
    只有按钮走一次性宏（独立触点）。
    """
    move: tuple[float, float] = (0.0, 0.0)
    button: str | None = None
    duration_ms: float = 200.0
    hold_ms: float = 40.0
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.button is not None and self.button not in ACTION_BUTTONS:
            raise ValueError(f"非法按钮: {self.button!r}（应为 {ACTION_BUTTONS}）")
        n = math.hypot(*self.move)
        if n > 1.0 + 1e-6:
            raise ValueError(f"move 需为单位向量或零向量, 模长={n}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength 需在 [0,1], 实为 {self.strength}")
        if self.hold_ms < 0.0:
            raise ValueError(f"hold_ms 不可为负, 实为 {self.hold_ms}")


# 离散动作表: 0=静止, 1-8=八方向; 9-13=按钮（OnnxPolicy 输出 argmax → Action）
DIRS_8: tuple[tuple[float, float], ...] = (
    (0.0, 0.0), (0.0, -1.0), (0.7071, -0.7071), (1.0, 0.0), (0.7071, 0.7071),
    (0.0, 1.0), (-0.7071, 0.7071), (-1.0, 0.0), (-0.7071, -0.7071),
)


def action_from_index(idx: int, duration_ms: float = 200.0) -> Action:
    if 0 <= idx <= 8:
        return Action(move=DIRS_8[idx], duration_ms=duration_ms)
    if 9 <= idx < 9 + len(ACTION_BUTTONS):
        return Action(button=ACTION_BUTTONS[idx - 9], duration_ms=duration_ms)
    raise ValueError(f"动作索引越界: {idx}")


# ═════════════════════════ 状态读取器 ═════════════════════════
class StateReader:
    """按 StateLayout 切片名+字段名读 64 维向量（决策层唯一合法取值方式）。"""

    def __init__(self, layout: StateLayout = DEFAULT_LAYOUT):
        self._idx: dict[tuple[str, str], int] = {}
        for s in layout.SLICES:
            for i, f in enumerate(s.fields):
                self._idx[(s.name, f)] = s.start + i

    def __call__(self, state: np.ndarray, slice_name: str, field: str) -> float:
        return float(state[self._idx[(slice_name, field)]])

    def seq(self, state: np.ndarray, slice_name: str) -> tuple[float, ...]:
        s = next(s for s in DEFAULT_LAYOUT.SLICES if s.name == slice_name)
        return tuple(float(x) for x in state[s.start:s.end])


# ═════════════════════════ 策略接口与实现 ═════════════════════════
class PolicyProvider(ABC):
    name: str = "policy"

    @abstractmethod
    def decide(self, state: np.ndarray) -> Action: ...


class RulePolicy(PolicyProvider):
    """规则策略 —— 64 维状态的 if-else 基线。

    优先级: 残血替身 > 满能量大招 > 近身技能1 > 走位(追击/拉开/待机)。
    """
    name = "rule"

    HP_SUB_THRESHOLD = 0.30
    ULT_ENERGY = 0.60
    SKILL_RANGE = 300.0        # 像素
    ENGAGE_MIN = 120.0         # 近身拉开阈值

    def __init__(self, reader: StateReader | None = None):
        self.r = reader or StateReader()

    def decide(self, state: np.ndarray) -> Action:
        r = self.r
        hp_self = r(state, "resources", "hp_self")
        sub_ready = r(state, "resources", "sub_ready_self")
        energy = r(state, "resources", "energy_self")
        ult_cd = r(state, "resources", "ult_cd_self")
        skill1_cd = r(state, "skill_cds", "skill1")
        enemy_visible = r(state, "enemy_kin", "visible") > 0.5

        # 1) 残血 + 替身可用 → 替身（保命最优先）
        if hp_self < self.HP_SUB_THRESHOLD and sub_ready > 0.5:
            return Action(button="sub")
        # 2) 能量达标 + 大招无 CD + 敌在 → 大招
        if energy >= self.ULT_ENERGY and ult_cd <= 0.01 and enemy_visible:
            return Action(button="ult")
        # 3) 敌在近身圈 + 技能1无 CD → 技能1
        dist = r(state, "relative", "dist") * 1500.0
        if enemy_visible and dist < self.SKILL_RANGE and skill1_cd <= 0.01:
            return Action(button="skill1")
        # 4) 走位: 追击 / 近身拉开 / 待机
        if enemy_visible:
            dx = (r(state, "enemy_kin", "x") - r(state, "self_kin", "x")) * 1280.0
            dy = (r(state, "enemy_kin", "y") - r(state, "self_kin", "y")) * 720.0
            n = math.hypot(dx, dy) or 1.0
            if dist < self.ENGAGE_MIN:           # 太近 → 反向拉开
                dx, dy = -dx, -dy
            return Action(move=(dx / n, dy / n))
        return Action(move=(0.0, 0.0))


class OnnxPolicy(PolicyProvider):
    """student.onnx FP16 前向 → argmax → 离散动作。真推理（M7 产物）。"""
    name = "onnx"

    def __init__(self, model_path: str = "student.onnx",
                 duration_ms: float = 200.0):
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError as e:
            raise DependencyMissing(
                "onnxruntime", "决策层 ONNX FP16 前向（M4.5+）",
                "pip install onnxruntime",
            ) from e
        self.duration_ms = duration_ms
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1              # 单核: 前向预算 5-15ms
        self.session = ort.InferenceSession(model_path, so,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def decide(self, state: np.ndarray) -> Action:
        x = state.astype(np.float32)[None, :]    # (1,64)
        out = self.session.run(None, {self.input_name: x})[0]
        idx = int(np.argmax(out[0]))
        return action_from_index(idx, self.duration_ms)


class SlowPolicy(PolicyProvider):
    """测试替身: 包一层固定延迟, 用于看门狗(超时沿用上帧动作)验证。"""
    name = "slow"

    def __init__(self, inner: PolicyProvider, delay_s: float):
        self.inner = inner
        self.delay_s = delay_s

    def decide(self, state: np.ndarray) -> Action:
        time.sleep(self.delay_s)
        return self.inner.decide(state)

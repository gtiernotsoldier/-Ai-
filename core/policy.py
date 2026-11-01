# -*- coding: utf-8 -*-
"""M4.5 — 决策层接口 (policy)

契约: PolicyProvider.decide(state: float32(64,)) -> Action
实现:
  RulePolicy  规则策略（读 64 维状态切片, 沙箱/联调/基线用）
  OnnxPolicy  student.onnx FP16 前向（四头/拼接/旧单头三种导出均兼容）
  SlowPolicy  测试替身（注入延迟 → 看门狗验证）

动作表达（M4.8, 对齐 Shūkai）:
  ActionHeads  四头 move_ud×move_lr×skill×direction —— 移动与技能可同时下发
  ActionMask   CD/能量/替身/被控 的合法性屏蔽
  ActionChunk  K 步意图块 → Sequencer 一次编译下发
  旧 action_from_index() 保留, 仅供旧 onnx 与历史测试。

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
from core.state import DEFAULT_LAYOUT, REL_POS_FULL_PX, StateLayout


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
    """旧单头动作表（17 选 1）。保留给旧 onnx / 旧测试, 新策略走 ActionHeads。"""
    if 0 <= idx <= 8:
        return Action(move=DIRS_8[idx], duration_ms=duration_ms)
    if 9 <= idx < 9 + len(ACTION_BUTTONS):
        return Action(button=ACTION_BUTTONS[idx - 9], duration_ms=duration_ms)
    raise ValueError(f"动作索引越界: {idx}")


# ═════════════════════════ 动作头（Shūkai 四头, ICML'24 2406.01103） ═════════════════════════
# 为什么不是单头 17 选 1: 单头把「移动」和「按钮」做成互斥, 训练出的策略永远
# 无法「边走边放技能」; 而 Action 数据类本来就支持 move+button 同时存在。
# 四头 = 上下(3) × 左右(3) × 技能(9=8键+空) × 施法朝向(9=8向+空)。
MOVE_AXIS: tuple[float, float, float] = (-1.0, 0.0, 1.0)
AXIS_IDLE = 1                                     # 中位

# 技能时长档位（秒→毫秒）: 训练标签与执行共用同一套档位, 避免连续值回归
DURATION_BUCKETS_MS: tuple[float, ...] = (40.0, 80.0, 120.0, 200.0, 320.0, 500.0)

# 大招最低能量（与 RulePolicy.ULT_ENERGY 同源; 动作屏蔽用它判合法性）
ULT_ENERGY_MIN = 0.60
# CD 视为「已就绪」的阈值（与 skill_cds 切片口径一致）
CD_READY_EPS = 0.01


def quantize_duration(ms: float) -> float:
    """把连续时长吸附到最近档位。训练标签与推理共用, 保证同一离散空间。"""
    return min(DURATION_BUCKETS_MS, key=lambda b: abs(b - float(ms)))


@dataclass(frozen=True)
class ActionHeads:
    """四头动作（策略输出 → 执行层 Action）。

    direction: 施法朝向。真机只有一个摇杆, 施法方向就是推杆方向 ——
      因此 direction>0 且 skill>0 时, 用它覆盖移动方向（施法瞬间的推杆）。
      direction=0 时按 move_ud/move_lr 组合走位。
    duration_ms: 这个决策要维持多久（档位量化）。M4.6 之后由 Sequencer 按帧表
      接管精确时刻, 这里表达的是「意图持续时长」。
    """
    move_ud: int = AXIS_IDLE
    move_lr: int = AXIS_IDLE
    skill: int = 0                    # 0=不放技能, 1..len(ACTION_BUTTONS)
    direction: int = 0                # 0=不指定, 1..8 = DIRS_8
    hold_ms: float = 40.0
    duration_ms: float = 200.0
    strength: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.move_ud <= 2:
            raise ValueError(f"move_ud 越界: {self.move_ud}")
        if not 0 <= self.move_lr <= 2:
            raise ValueError(f"move_lr 越界: {self.move_lr}")
        if not 0 <= self.skill <= len(ACTION_BUTTONS):
            raise ValueError(f"skill 越界: {self.skill}")
        if not 0 <= self.direction <= 8:
            raise ValueError(f"direction 越界: {self.direction}")

    @property
    def button(self) -> str | None:
        return ACTION_BUTTONS[self.skill - 1] if self.skill > 0 else None

    def move(self) -> tuple[float, float]:
        if self.skill > 0 and self.direction > 0:
            return DIRS_8[self.direction]          # 施法瞬间的推杆方向
        x = MOVE_AXIS[self.move_lr]
        y = MOVE_AXIS[self.move_ud]
        n = math.hypot(x, y)
        return (x / n, y / n) if n > 0 else (0.0, 0.0)

    def to_action(self) -> Action:
        return Action(move=self.move(), button=self.button,
                      duration_ms=quantize_duration(self.duration_ms),
                      hold_ms=self.hold_ms, strength=self.strength)


@dataclass(frozen=True)
class ActionMask:
    """动作合法性屏蔽（Shūkai 的 action mask）。

    CD 没好 / 能量不够 / 替身不可用 / 被控倒地 —— 这些动作在游戏里根本发不出去,
    让策略在这些位置上探索纯属浪费样本。屏蔽只表达「合法性」, 不含策略偏好。
    """
    move_ud: tuple[bool, bool, bool]
    move_lr: tuple[bool, bool, bool]
    skill: tuple[bool, ...]
    direction: tuple[bool, ...]

    @classmethod
    def from_state(cls, state: np.ndarray,
                   reader: "StateReader | None" = None) -> "ActionMask":
        r = reader or StateReader()
        cc = (r(state, "effects", "cc") > 0.5
              or r(state, "effects", "knockdown") > 0.5)
        if cc:
            # 被控/倒地: 只能回中停手（连摇杆都推不动）
            idle_move = (False, True, False)
            return cls(idle_move, idle_move,
                       (True,) + (False,) * len(ACTION_BUTTONS),
                       (True,) + (False,) * 8)

        ok_skill = [True]                            # 0 = 不放技能, 永远合法
        for b in ACTION_BUTTONS:
            if b == "sub":
                ok = r(state, "resources", "sub_ready_self") > 0.5
            elif b == "ult":
                ok = (r(state, "skill_cds", "ult") <= CD_READY_EPS
                      and r(state, "resources", "energy_self") >= ULT_ENERGY_MIN)
            elif b == "dash":
                ok = r(state, "self_kin", "dash_cd") <= 0.05
            elif b in ("skill1", "skill2", "skill3"):
                ok = r(state, "skill_cds", b) <= CD_READY_EPS
            else:                                    # atk / secret: 无 CD 门
                ok = True
            ok_skill.append(ok)
        return cls((True, True, True), (True, True, True),
                   tuple(ok_skill), tuple([True] * 9))

    def skill_logits(self, logits) -> np.ndarray:
        """掩码后的技能 logits（被屏蔽位置置 -1e9）。"""
        arr = np.asarray(logits, dtype=np.float32).copy()
        allowed = np.asarray(self.skill, dtype=bool)
        arr[~allowed] = -1e9
        return arr


# 意图分块: 20Hz 一次决策, 一次给 K 步（K×50ms 视界）。
# 作用不是省算力, 是让连段/走位在时间上连贯 —— 人不会每 50ms 重想一次。
MAX_CHUNK = 8
DEFAULT_CHUNK = 4


@dataclass(frozen=True)
class ActionChunk:
    """K 步意图块（BC/PPO 的输出单位, Sequencer 的输入单位）。

    走位不进序列: 摇杆由 JoystickStream 持久流承载, 每 tick 在线更新;
    本块只负责「按钮序列 + 每步按压时长」, 由 Sequencer 一次编译下发,
    中途遇事件边沿（如 enemy_sub）仍可帧级打断。
    """
    heads: tuple[ActionHeads, ...]

    def __post_init__(self) -> None:
        if not self.heads:
            raise ValueError("ActionChunk 至少需要一步")
        if len(self.heads) > MAX_CHUNK:
            raise ValueError(f"ActionChunk 最多 {MAX_CHUNK} 步, 实为 {len(self.heads)}")

    @property
    def duration_ms(self) -> float:
        return sum(h.duration_ms for h in self.heads)

    def to_sequence(self, name: str = "chunk"):
        """转成 Sequencer 的 Sequence（只含按钮步; 全走位块会报错）。"""
        from core.sequencer import Sequence, Step      # 局部导入: 避免模块环
        steps = tuple(Step(h.button, hold_ms=h.hold_ms)
                      for h in self.heads if h.button is not None)
        if not steps:
            raise ValueError("该 chunk 没有任何按钮动作, 不需要提交序列")
        return Sequence(name, steps)


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
            # 契约 v3: enemy_kin.dx/dy 已是「敌方相对自机」的位移(px),
            # 不再用两侧绝对坐标相减 —— 镜头滚动不会改变这个量。
            dx = (r(state, "enemy_kin", "dx") - 0.5) * 2 * REL_POS_FULL_PX
            dy = (r(state, "enemy_kin", "dy") - 0.5) * 2 * REL_POS_FULL_PX
            n = math.hypot(dx, dy) or 1.0
            if dist < self.ENGAGE_MIN:           # 太近 → 反向拉开
                dx, dy = -dx, -dy
            return Action(move=(dx / n, dy / n))
        return Action(move=(0.0, 0.0))


class OnnxPolicy(PolicyProvider):
    """student.onnx FP16 前向（M7 产物）。

    支持三种导出形态, 按输出形状自动识别:
      1. 四头独立输出      → [move_ud(3), move_lr(3), skill(9), direction(9)]
      2. 单输出拼接        → (1, 24) 按同样顺序切分
      3. 旧单头            → (1, 17) 走 action_from_index（向后兼容）
    """
    name = "onnx"

    HEAD_SIZES: tuple[int, ...] = (3, 3, len(ACTION_BUTTONS) + 1, 9)
    TOTAL_HEAD_LOGITS = sum(HEAD_SIZES)          # 24

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
        self._reader = StateReader()             # 复用切片索引表, 不 per-tick 重建

    @staticmethod
    def _argmax(x) -> int:
        return int(np.argmax(np.asarray(x).reshape(-1)))

    def _decode(self, outs: list, mask: ActionMask) -> Action:
        if len(outs) >= 4:
            ud, lr, sk, dr = outs[:4]
            return ActionHeads(
                move_ud=self._argmax(ud), move_lr=self._argmax(lr),
                skill=self._argmax(mask.skill_logits(sk[0])),
                direction=self._argmax(dr),
                duration_ms=self.duration_ms).to_action()
        flat = np.asarray(outs[0]).reshape(-1)
        if flat.size == self.TOTAL_HEAD_LOGITS:      # 单输出拼接
            a, b = 0, self.HEAD_SIZES[0]
            ud = flat[a:b]; a, b = b, b + self.HEAD_SIZES[1]
            lr = flat[a:b]; a, b = b, b + self.HEAD_SIZES[2]
            sk = flat[a:b]; a, b = b, b + self.HEAD_SIZES[3]
            dr = flat[a:b]
            return ActionHeads(
                move_ud=self._argmax(ud), move_lr=self._argmax(lr),
                skill=self._argmax(mask.skill_logits(sk)),
                direction=self._argmax(dr),
                duration_ms=self.duration_ms).to_action()
        # 旧单头: 17 选 1
        return action_from_index(self._argmax(flat), self.duration_ms)

    def decide(self, state: np.ndarray) -> Action:
        x = state.astype(np.float32)[None, :]    # (1,64)
        outs = self.session.run(None, {self.input_name: x})
        mask = ActionMask.from_state(state, self._reader)
        return self._decode(outs, mask)


class SlowPolicy(PolicyProvider):
    """测试替身: 包一层固定延迟, 用于看门狗(超时沿用上帧动作)验证。"""
    name = "slow"

    def __init__(self, inner: PolicyProvider, delay_s: float):
        self.inner = inner
        self.delay_s = delay_s

    def decide(self, state: np.ndarray) -> Action:
        time.sleep(self.delay_s)
        return self.inner.decide(state)

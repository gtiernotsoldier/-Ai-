# -*- coding: utf-8 -*-
"""M2/M3 — 感知层 (perception)

三路信息:
  快变量 60Hz — FastDetector (YOLOv8n / Mock): 角色/特效坐标
  慢变量 12Hz — CVReader (纯 OpenCV): 血条、能量、CD 读数（5帧一读）
  事件计时器  — EventTimer: 替身/大招 CD 靠特效帧起表推算（人类同款方法）

全部读数为真实图像计算（颜色掩码 + 几何占比），不依赖合成帧真值；
合成帧仅提供"带真值的测试画面"。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from core.constants import ENEMY_POS_TAU_S, EVENT_CD_S
from core.detectors import Detection, FastDetector
from core.keymap import Keymap
from core.screen_capture import Frame


# ═════════════════════════ 事件计时器 ═════════════════════════
class EventTimer:
    """特效帧起表推算 CD（如敌方替身闪特效 → 40s 不可再骗）。

    clock 可注入, 单测无需真实等待。
    """

    def __init__(self, duration_s: float, clock: Callable[[], float] = time.monotonic):
        self.duration = duration_s
        self._clock = clock
        self._start: float | None = None

    def start(self) -> None:
        self._start = self._clock()

    def remaining(self) -> float:
        if self._start is None:
            return 0.0
        r = self.duration - (self._clock() - self._start)
        return max(0.0, r)

    @property
    def running(self) -> bool:
        return self._start is not None and self.remaining() > 0.0


# ═════════════════════════ OpenCV 慢变量读数 ═════════════════════════
def _color_fraction(img: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """区域内命中颜色掩码的像素占比。"""
    mask = cv2.inRange(img, lower, upper)
    return float(np.count_nonzero(mask)) / mask.size


class CVReader:
    """血条/能量/CD 的纯 OpenCV 读数。所有比值 ∈ [0,1]。"""

    # 掩码阈值（BGR）
    _HP_LOWER, _HP_UPPER = np.array([30, 140, 30]), np.array([110, 240, 110])
    _EN_LOWER, _EN_UPPER = np.array([200, 140, 30]), np.array([255, 240, 110])
    _CD_LOWER, _CD_UPPER = np.array([50, 24, 8]), np.array([82, 64, 55])   # 上限含扇形边缘AA混色, 但排除灰底(90,90,90)
    _SUB_LOWER, _SUB_UPPER = np.array([0, 200, 200]), np.array([90, 255, 255])  # 黄=BGR(0,255,255)

    def __init__(self, keymap: Keymap):
        self.km = keymap

    def hp_ratio(self, img: np.ndarray, side: str) -> float:
        x, y, w, h = self.km.rect(f"hp_{side}")
        return _color_fraction(img[y:y + h, x:x + w], self._HP_LOWER, self._HP_UPPER)

    def energy_ratio(self, img: np.ndarray, side: str) -> float:
        x, y, w, h = self.km.rect(f"energy_{side}")
        return _color_fraction(img[y:y + h, x:x + w], self._EN_LOWER, self._EN_UPPER)

    def cd_remaining(self, img: np.ndarray, btn: str) -> float:
        """剩余 CD: 内接圆盘内扇形遮罩面积占比 = 角度占比（真实几何对应）。

        与渲染端 draw_pie 的半径 r-4 严格对齐, 并用圆盘掩码
        排除按钮外环的抗锯齿混色像素。
        """
        cx, cy, r = self.km.circle(f"btn_{btn}")
        rr = r - 4                                   # 与 draw_pie 半径一致
        roi = img[cy - r:cy + r, cx - r:cx + r]
        disk = np.zeros(roi.shape[:2], np.uint8)
        cv2.circle(disk, (r, r), rr, 255, -1)        # 半径与 draw_pie 一致, 覆盖最外圈AA像素
        mask = cv2.inRange(roi, self._CD_LOWER, self._CD_UPPER)
        count = int(np.count_nonzero(cv2.bitwise_and(mask, disk)))
        return float(np.clip(count / (math.pi * rr * rr), 0.0, 1.0))

    def sub_ready(self, img: np.ndarray) -> bool:
        cx, cy, r = self.km.circle("btn_sub")
        roi = img[cy - r:cy + r, cx - r:cx + r]
        disk = np.zeros(roi.shape[:2], np.uint8)
        cv2.circle(disk, (r, r), r, 255, -1)
        mask = cv2.inRange(roi, self._SUB_LOWER, self._SUB_UPPER)
        # 显式 bool(): 否则返回 np.bool_, numpy>=2.5 下 `is True` 判定会失败,
        # 且 numpy 标量会顺着 PerceptionResult 渗进状态/决策层。
        return bool(np.count_nonzero(cv2.bitwise_and(mask, disk)) > 40)


# cv2 延迟绑定（保持模块可被无 GUI 环境导入）
import cv2  # noqa: E402


# ═════════════════════════ 感知结果 ═════════════════════════
@dataclass
class PerceptionResult:
    frame_id: int
    ts: float
    # 快变量
    self_pos: tuple[float, float] = (0.0, 0.0)
    enemy_pos: tuple[float, float] = (0.0, 0.0)
    self_vel: tuple[float, float] = (0.0, 0.0)
    enemy_vel: tuple[float, float] = (0.0, 0.0)
    enemy_visible: bool = False
    enemy_pos_confidence: float = 0.0     # P2#4: 失踪后 exp(-age/τ) 衰减
    closing_speed: float = 0.0            # px/s, >0 = 双方接近中
    # 慢变量（缓存, 12Hz 刷新）
    hp_self: float = 0.0
    hp_enemy: float = 0.0
    energy_self: float = 0.0
    energy_enemy: float = 0.0             # HUD 不可见, 恒 0（公式先就位, 检测器扩类后有数即用）
    sub_ready_self: bool = False
    sub_ready_enemy: bool = True          # 打磨清单①: 敌方替身就绪(事件计时器推算)
    cds: dict = field(default_factory=dict)
    # 事件计时器
    event_remaining: dict = field(default_factory=dict)
    # 事件边沿(打磨清单⑦): 本帧新触发的事件名 —— Sequencer 帧级分支的触发源
    events_fired: tuple[str, ...] = ()


# ═════════════════════════ 感知主类 ═════════════════════════
class Perception:
    """60Hz 快变量 + 12Hz 慢变量 + 事件计时器 → PerceptionResult。

    实战闭环内仅做内存计算（宪法: 零网络/零磁盘/零睡眠）。
    """

    def __init__(self, fast: FastDetector, cv: CVReader,
                 slow_every: int = 5,
                 clock: Callable[[], float] = time.monotonic):
        self.fast = fast
        self.cv = cv
        self.slow_every = slow_every
        self._clock = clock
        self._prev_pos: dict[str, tuple[float, float]] = {}
        self._prev_ts: float | None = None
        self._slow_cache: dict = {}
        self._events: dict[str, EventTimer] = {}
        # 打磨清单①: 生产路径注册事件计时器（此前 register_event 从未被调用,
        # trigger_event 静默 no-op, event_remaining 永远空 dict）。
        # 敌方替身/大招 HUD 不可见 → 只能靠特效帧起表推算。
        for name, dur in EVENT_CD_S.items():
            self.register_event(name, dur)
        # P2#4: 敌人失踪时保留最后已知位置 + 置信度随时间衰减
        self._last_enemy_pos: tuple[float, float] | None = None
        self._last_enemy_seen_ts: float | None = None

    # ── 事件计时器管理 ──
    def register_event(self, name: str, duration_s: float) -> None:
        self._events[name] = EventTimer(duration_s, self._clock)

    def trigger_event(self, name: str) -> None:
        """特效帧出现 → 起表。如看到敌方替身闪 → trigger_event("enemy_sub")。"""
        if name in self._events:
            self._events[name].start()

    # ── 主更新（每帧调用, 目标 60Hz）──
    def update(self, frame: Frame) -> PerceptionResult:
        now = frame.ts
        # 快变量: 每帧
        dets = self.fast.detect(frame)
        pos: dict[str, tuple[float, float]] = {}
        for d in dets:
            if d.cls in ("self", "enemy"):
                pos[d.cls] = (d.x + d.w / 2, d.y + d.h / 2)
        vel: dict[str, tuple[float, float]] = {"self": (0.0, 0.0), "enemy": (0.0, 0.0)}
        if self._prev_ts is not None and now > self._prev_ts:
            dt = now - self._prev_ts
            for k, p in pos.items():
                if k in self._prev_pos:
                    vel[k] = ((p[0] - self._prev_pos[k][0]) / dt,
                              (p[1] - self._prev_pos[k][1]) / dt)
        self._prev_pos, self._prev_ts = pos, now

        # P2#4: 敌人本帧不可见 → 保留最后已知位置, 置信度 exp(-age/τ) 衰减
        if "enemy" in pos:
            self._last_enemy_pos = pos["enemy"]
            self._last_enemy_seen_ts = now
        enemy_pos = pos.get("enemy", self._last_enemy_pos or (0.0, 0.0))
        if self._last_enemy_pos is not None and self._last_enemy_seen_ts is not None:
            age = max(0.0, now - self._last_enemy_seen_ts)
            enemy_conf = float(math.exp(-age / ENEMY_POS_TAU_S))
        else:
            enemy_conf = 0.0

        # 接近速度: 相对速度在视线方向上的投影, >0 = 接近中
        sp = pos.get("self", (0.0, 0.0))
        dx, dy = enemy_pos[0] - sp[0], enemy_pos[1] - sp[1]
        dist = math.hypot(dx, dy)
        if dist > 1e-3:
            closing = ((vel["self"][0] - vel["enemy"][0]) * dx
                       + (vel["self"][1] - vel["enemy"][1]) * dy) / dist
        else:
            closing = 0.0

        # 特效 → 事件起表（打磨清单①: 计时器已在 __init__ 注册, 不再是 no-op）
        # 同时记录边沿(fired only on rising edge of this frame) → 主循环转交
        # Sequencer.on_events() 做帧级分支, 免去策略层 20Hz 轮询（打磨清单⑦）
        fired: list[str] = []
        if any(d.cls == "effect_sub" for d in dets):
            self.trigger_event("enemy_sub")
            fired.append("enemy_sub")
        if any(d.cls == "effect_ult" for d in dets):
            self.trigger_event("enemy_ult")
            fired.append("enemy_ult")

        # 慢变量: 5帧一读（60Hz→12Hz）
        if frame.frame_id % self.slow_every == 0:
            img = frame.image
            self._slow_cache = {
                "hp_self": self.cv.hp_ratio(img, "self"),
                "hp_enemy": self.cv.hp_ratio(img, "enemy"),
                "energy_self": self.cv.energy_ratio(img, "self"),
                "sub_ready_self": self.cv.sub_ready(img),
                "cds": {b: self.cv.cd_remaining(img, b)
                        for b in ("skill1", "skill2", "skill3", "ult", "sub")},
            }

        event_remaining = {k: t.remaining() for k, t in self._events.items()}
        # 敌方替身就绪 = 计时器不在走（没见过特效或已走完 → 可骗/可用）
        sub_ready_enemy = True
        if "enemy_sub" in self._events:
            sub_ready_enemy = not self._events["enemy_sub"].running

        res = PerceptionResult(
            frame_id=frame.frame_id,
            ts=now,
            self_pos=pos.get("self", (0.0, 0.0)),
            enemy_pos=enemy_pos,
            self_vel=vel["self"],
            enemy_vel=vel["enemy"],
            enemy_visible="enemy" in pos,
            enemy_pos_confidence=enemy_conf,
            closing_speed=float(closing),
            hp_self=self._slow_cache.get("hp_self", 0.0),
            hp_enemy=self._slow_cache.get("hp_enemy", 0.0),
            energy_self=self._slow_cache.get("energy_self", 0.0),
            sub_ready_self=self._slow_cache.get("sub_ready_self", False),
            sub_ready_enemy=sub_ready_enemy,
            cds=dict(self._slow_cache.get("cds", {})),
            event_remaining=event_remaining,
            events_fired=tuple(fired),
        )
        return res

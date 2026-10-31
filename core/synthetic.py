# -*- coding: utf-8 -*-
"""合成对局画面生成器（开发/单测专用，不进入实战闭环）。

用 OpenCV 画出带已知真值(Ground Truth)的"火影手游风格"帧：
角色、血条、能量条、技能按钮(含 CD 扇形遮罩)、替身图标。
CVReader 在这些帧上做的读数是真实图像计算，不是查表——
因此沙箱单测通过 ≈ 感知算法本身正确，只差真机标定(keymap)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.constants import FRAME_W, FRAME_H

# BGR 调色板
_BG_COLOR = (38, 30, 24)          # 暗色战场底
_TRACK_GRAY = (60, 60, 60)        # 条/按钮空槽
_HP_GREEN = (63, 191, 63)         # 血条填充
_ENERGY_BLUE = (255, 191, 63)     # 能量条填充
_BTN_READY = (0, 153, 255)        # 按钮就绪(亮橙)
_BTN_BASE = (90, 90, 90)          # 按钮底
_CD_OVERLAY = (64, 32, 16)        # CD 剩余扇形遮罩(深藏青)
_SUB_RING = (0, 255, 255)         # 替身就绪黄环
_SELF_BLUE = (255, 160, 60)       # 己方角色
_ENEMY_RED = (60, 60, 230)        # 敌方角色
_PROJ_CYAN = (255, 255, 0)        # ⑧ 飞行道具
_WARN_RED = (0, 0, 220)           # ⑧ 技能预警圈
_WALL_GRAY = (110, 80, 50)        # ⑧ 场地墙体


@dataclass(frozen=True)
class Projectile:
    """⑧ 飞行道具真值: 位置 + 速度(px/s)。Mock 检测器逐帧读它, 真机由 YOLO 逐帧框出。"""
    pos: tuple[float, float]
    vel: tuple[float, float]


@dataclass(frozen=True)
class HitboxWarn:
    """⑧ 技能预警圈真值: 圆心 + 半径（范围提示, 判定尚未生效）。"""
    pos: tuple[float, float]
    radius: float


@dataclass
class Scene:
    """一帧的全部真值。测试 = 写 Scene → 渲染 → CV 读数 → 对账。"""
    self_pos: tuple[float, float] = (400.0, 500.0)
    enemy_pos: tuple[float, float] = (800.0, 400.0)
    self_vel: tuple[float, float] = (0.0, 0.0)
    enemy_vel: tuple[float, float] = (0.0, 0.0)
    facing_self: float = 0.0            # 弧度
    facing_enemy: float = math.pi       # ⑧ 敌方朝向(弧度); 默认朝左（面向己方）
    hp_self: float = 1.0                # [0,1]
    hp_enemy: float = 1.0
    energy_self: float = 0.5
    energy_enemy: float = 0.5
    sub_ready_self: bool = True
    sub_ready_enemy: bool = True
    cds: dict = field(default_factory=lambda: {
        "skill1": 0.0, "skill2": 0.0, "skill3": 0.0,
        "ult": 0.0, "sub": 0.0,
    })
    effects: set = field(default_factory=set)   # "hit_flash" 等
    # ⑧ 检测器扩类: 场上动态实体真值
    projectiles: tuple = field(default_factory=tuple)     # tuple[Projectile, ...]
    hitbox_warns: tuple = field(default_factory=tuple)    # tuple[HitboxWarn, ...]
    walls: tuple = field(default_factory=tuple)           # tuple[(x,y,w,h), ...] 场地墙段


def draw_pie(img, cx: int, cy: int, r: int, fraction: float, color) -> None:
    """画剩余 CD 扇形遮罩: fraction=剩余比例, 面积占比=角度占比（真实几何）。"""
    if fraction <= 0.003:
        return
    angle = max(1, int(360.0 * min(fraction, 1.0)))
    cv2.ellipse(img, (cx, cy), (r - 4, r - 4), 0, -90, -90 + angle,
                color, thickness=-1, lineType=cv2.LINE_AA)


def render(scene: Scene, width: int = FRAME_W, height: int = FRAME_H) -> np.ndarray:
    img = np.full((height, width, 3), _BG_COLOR, dtype=np.uint8)

    # ── 血条(顶部) / 能量条(左下) ──
    for x, ratio, color in ((40, scene.hp_self, _HP_GREEN),
                            (940, scene.hp_enemy, _HP_GREEN)):
        cv2.rectangle(img, (x, 30), (x + 300, 48), _TRACK_GRAY, -1)
        w = int(round(300 * min(max(ratio, 0.0), 1.0)))
        if w > 0:
            cv2.rectangle(img, (x, 30), (x + w, 48), color, -1)
    cv2.rectangle(img, (40, 650), (340, 660), _TRACK_GRAY, -1)
    ew = int(round(300 * min(max(scene.energy_self, 0.0), 1.0)))
    if ew > 0:
        cv2.rectangle(img, (40, 650), (40 + ew, 660), _ENERGY_BLUE, -1)

    # ── 技能按钮: 就绪亮橙 / CD 画剩余扇形遮罩 ──
    buttons = {"skill1": (1050, 640, 30), "skill2": (1130, 640, 30),
               "skill3": (1210, 640, 30), "ult": (1150, 540, 36),
               "sub": (980, 540, 30)}
    for name, (cx, cy, r) in buttons.items():
        cv2.circle(img, (cx, cy), r, _BTN_BASE, -1, lineType=cv2.LINE_AA)
        cd = scene.cds.get(name, 0.0)
        if cd > 0.003:
            draw_pie(img, cx, cy, r, cd, _CD_OVERLAY)
        else:
            cv2.circle(img, (cx, cy), r - 6, _BTN_READY, -1, lineType=cv2.LINE_AA)
    # 替身就绪黄环（CV 通过黄色像素量判就绪）
    if scene.sub_ready_self:
        cv2.circle(img, (980, 540), 30, _SUB_RING, 3, lineType=cv2.LINE_AA)

    # ── ⑧ 场地墙体（YOLO wall 类的合成对齐物）──
    for (wx, wy, ww, wh) in scene.walls:
        cv2.rectangle(img, (int(wx), int(wy)), (int(wx + ww), int(wy + wh)),
                      _WALL_GRAY, -1)

    # ── ⑧ 技能预警圈 / 飞行道具 ──
    for hb in scene.hitbox_warns:
        cv2.circle(img, (int(hb.pos[0]), int(hb.pos[1])), int(hb.radius),
                   _WARN_RED, 3, lineType=cv2.LINE_AA)
    for pj in scene.projectiles:
        cv2.circle(img, (int(pj.pos[0]), int(pj.pos[1])), 8,
                   _PROJ_CYAN, -1, lineType=cv2.LINE_AA)

    # ── 角色 ──
    sx, sy = int(scene.self_pos[0]), int(scene.self_pos[1])
    ex, ey = int(scene.enemy_pos[0]), int(scene.enemy_pos[1])
    cv2.circle(img, (sx, sy), 30, _SELF_BLUE, -1, lineType=cv2.LINE_AA)
    fx = int(30 * math.cos(scene.facing_self))
    fy = int(30 * math.sin(scene.facing_self))
    cv2.arrowedLine(img, (sx, sy), (sx + fx, sy + fy), (255, 255, 255), 3)
    cv2.circle(img, (ex, ey), 30, _ENEMY_RED, -1, lineType=cv2.LINE_AA)
    # ⑧ 敌方朝向箭头（绕后 flank 判定的感知来源）
    efx = int(30 * math.cos(scene.facing_enemy))
    efy = int(30 * math.sin(scene.facing_enemy))
    cv2.arrowedLine(img, (ex, ey), (ex + efx, ey + efy), (0, 0, 0), 3)

    return img

# -*- coding: utf-8 -*-
"""快变量检测器（可插拔）。

接口契约: detect(frame) -> list[Detection]
实现:
  MockFastDetector — 从合成帧 Scene 真值直接生成检测结果（沙箱/CI 用）
  YoloFastDetector — ultralytics YOLOv8n 真推理（真机用, 缺库优雅报错）
"""
from __future__ import annotations

from dataclasses import dataclass

from core.exceptions import DependencyMissing
from core.screen_capture import Frame


@dataclass(frozen=True)
class Detection:
    cls: str            # "self" | "enemy" | "effect_sub" | "projectile" | ...
    x: float            # 像素坐标
    y: float
    w: float
    h: float
    conf: float
    facing: float | None = None   # ⑧ 角色朝向(弧度), 仅 cls=self/enemy 时有意义


class FastDetector:
    def detect(self, frame: Frame) -> list[Detection]:
        raise NotImplementedError


class MockFastDetector(FastDetector):
    """直接读取合成帧携带的 Scene 真值 → 产出与 YOLO 同构的 Detection。

    用途: 在没有 GPU/ultralytics 的环境验证感知→状态全链路。
    ⑧ 扩类后同步产出: projectile / hitbox_warn / wall / 敌我朝向 / 特效标志。
    """

    def detect(self, frame: Frame) -> list[Detection]:
        if frame.scene is None:
            return []
        s = frame.scene
        dets = [
            Detection("self", s.self_pos[0] - 30, s.self_pos[1] - 30, 60, 60,
                      0.99, facing=s.facing_self),
            Detection("enemy", s.enemy_pos[0] - 30, s.enemy_pos[1] - 30, 60, 60,
                      0.99, facing=s.facing_enemy),
        ]
        if "sub_flash_enemy" in s.effects:
            dets.append(Detection("effect_sub", s.enemy_pos[0] - 40,
                                  s.enemy_pos[1] - 40, 80, 80, 0.9))
        # ⑧: 飞行道具 / 技能预警圈 / 场地墙
        for pj in s.projectiles:
            dets.append(Detection("projectile", pj.pos[0] - 8, pj.pos[1] - 8,
                                  16, 16, 0.95))
        for hb in s.hitbox_warns:
            dets.append(Detection("hitbox_warn",
                                  hb.pos[0] - hb.radius, hb.pos[1] - hb.radius,
                                  hb.radius * 2, hb.radius * 2, 0.9))
        for (wx, wy, ww, wh) in s.walls:
            dets.append(Detection("wall", wx, wy, ww, wh, 0.95))
        # ⑧: 自身状态特效 → 独立检测类（YOLO 训练后同理）
        for eff in sorted(s.effects):
            if eff in _SELF_EFFECT_NAMES:
                dets.append(Detection(f"effect_{eff}",
                                      s.self_pos[0] - 25, s.self_pos[1] - 25,
                                      50, 50, 0.9))
        return dets


# ⑧ 自身状态特效名 → Detection 类名前缀 effect_*; 状态向量 effects 切片同名字段
# （hit_flash/i_frame/knockdown/cc/super_armor 为自身状态; sub_flash_self 敌我替身区分, 清单①）
_SELF_EFFECT_NAMES = frozenset(
    {"hit_flash", "i_frame", "knockdown", "cc", "super_armor", "sub_flash_self"})


class YoloFastDetector(FastDetector):
    """YOLOv8n 真推理。classes 映射见 keymap/docs（M2 标定后定稿）。"""

    # 类别id → 语义（⑧ 扩类: 0-3 原有; 4-6 新增实体类; 7-12 自身状态特效类）
    CLASS_NAMES = {
        0: "self", 1: "enemy", 2: "effect_sub", 3: "effect_ult",
        4: "projectile", 5: "hitbox_warn", 6: "wall",
        7: "effect_hit_flash", 8: "effect_i_frame", 9: "effect_knockdown",
        10: "effect_cc", 11: "effect_super_armor", 12: "effect_sub_flash_self",
    }

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35,
                 input_size: int = 320):
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError as e:
            raise DependencyMissing(
                "ultralytics", "YOLOv8n 快变量检测（角色坐标/特效分类）",
                "pip install ultralytics",
            ) from e
        self.input_size = input_size
        self.conf = conf
        self.model = __import__("ultralytics", fromlist=["YOLO"]).YOLO(weights)

    def detect(self, frame: Frame) -> list[Detection]:
        results = self.model.predict(
            frame.image[:, :, ::-1],            # BGR→RGB
            imgsz=self.input_size,
            conf=self.conf,
            verbose=False,
        )[0]
        out: list[Detection] = []
        for b in results.boxes:
            cls = self.CLASS_NAMES.get(int(b.cls.item()), f"cls{int(b.cls.item())}")
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append(Detection(cls, x1, y1, x2 - x1, y2 - y1,
                                 float(b.conf.item())))
        return out

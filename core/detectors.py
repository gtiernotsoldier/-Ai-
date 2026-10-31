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
    cls: str            # "self" | "enemy" | "effect_sub" | ...
    x: float            # 像素坐标
    y: float
    w: float
    h: float
    conf: float


class FastDetector:
    def detect(self, frame: Frame) -> list[Detection]:
        raise NotImplementedError


class MockFastDetector(FastDetector):
    """直接读取合成帧携带的 Scene 真值 → 产出与 YOLO 同构的 Detection。

    用途: 在没有 GPU/ultralytics 的环境验证感知→状态全链路。
    """

    def detect(self, frame: Frame) -> list[Detection]:
        if frame.scene is None:
            return []
        s = frame.scene
        dets = [
            Detection("self", s.self_pos[0] - 30, s.self_pos[1] - 30, 60, 60, 0.99),
            Detection("enemy", s.enemy_pos[0] - 30, s.enemy_pos[1] - 30, 60, 60, 0.99),
        ]
        if "sub_flash_enemy" in s.effects:
            dets.append(Detection("effect_sub", s.enemy_pos[0] - 40,
                                  s.enemy_pos[1] - 40, 80, 80, 0.9))
        return dets


class YoloFastDetector(FastDetector):
    """YOLOv8n 真推理。classes 映射见 keymap/docs（M2 标定后定稿）。"""

    # 类别id → 语义
    CLASS_NAMES = {0: "self", 1: "enemy", 2: "effect_sub", 3: "effect_ult"}

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

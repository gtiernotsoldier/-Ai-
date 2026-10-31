# -*- coding: utf-8 -*-
"""M1 — 画面捕获 (scrcpy 流)

设计:
  FrameSource 抽象 ──┬── ScrcpySource     真机: scrcpy 裸 H.264 流, 需 scrcpy+av, 缺依赖时优雅报错
                     └── SyntheticSource  开发: 合成对局帧, 沙箱/CI 全功能可跑

宪法对齐:
  - 弃 ADB 截图（仅 8FPS），只走 scrcpy 视频流（预算 10ms）
  - 原始画面即焚: Frame 只在闭环内传递, 不提供任何落盘方法
"""
from __future__ import annotations

import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.constants import FRAME_W, FRAME_H
from core.exceptions import DependencyMissing
from core.synthetic import Scene, render


@dataclass
class Frame:
    """一帧画面。image 为 BGR uint8 (H, W, 3)。scene 仅供合成源携带真值。"""
    image: np.ndarray
    ts: float                    # time.monotonic() 捕获时刻
    frame_id: int
    source_fps: float
    scene: Optional[Scene] = field(default=None, repr=False)


class FrameSource(ABC):
    """帧源接口。所有实现必须非阻塞友好（read 返回最新帧）。"""

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def read(self) -> Frame: ...


# ═════════════════════════ 真机实现 ═════════════════════════
class ScrcpySource(FrameSource):
    """scrcpy 裸流捕获。

    管线: adb push scrcpy-server → app 进程编码 H.264 → 本地 socket
          → av(PyAV) 解码 → BGR ndarray
    依赖: scrcpy 二进制 + `pip install av`（二者任缺 → DependencyMissing）
    60FPS 典型端到端延迟 10ms 内, 符合延迟预算表。
    """

    def __init__(self, serial: Optional[str] = None,
                 width: int = FRAME_W, height: int = FRAME_H, max_fps: int = 60):
        self.serial = serial
        self.width, self.height, self.max_fps = width, height, max_fps
        self._proc = None
        self._av = None
        self._container = None
        self._frame_id = 0
        self._t0 = 0.0
        self._last_ts: list[float] = []

    def open(self) -> None:
        if shutil.which("scrcpy") is None:
            raise DependencyMissing(
                "scrcpy", "真机画面捕获（M1）", "Windows: winget install scrcpy / choco install scrcpy"
            )
        try:
            import av  # noqa: F401
        except ImportError as e:
            raise DependencyMissing("av", "解码 scrcpy H.264 裸流", "pip install av") from e
        # 具体管线在真机环境首次联调时启用（沙箱无 adb 设备, 不进入此分支）:
        #   cmd = ["scrcpy", "--no-window", "--no-audio", "--max-fps", str(self.max_fps),
        #          "--video-codec=h264", "--raw-stream=-"]  # 裸流 → stdout
        raise DependencyMissing(
            "scrcpy-pipeline",
            "真机管线需 adb 设备在场联调（沙箱/开发机请用 SyntheticSource）",
            "接真机后按 docs 内管线启用",
        )

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()

    def read(self) -> Frame:  # pragma: no cover - 真机联调路径
        raise DependencyMissing("scrcpy-pipeline", "见 open()", "-")


# ═════════════════════════ 开发/测试实现 ═════════════════════════
class SyntheticSource(FrameSource):
    """合成对局帧源。可携带 Scene 真值, 供感知层对账与延迟基准。"""

    def __init__(self, fps: int = 60,
                 width: int = FRAME_W, height: int = FRAME_H,
                 scene: Optional[Scene] = None):
        self.fps = fps
        self.width, self.height = width, height
        self.scene = scene or Scene()
        self._frame_id = 0
        self._t0 = 0.0
        self._opened = False

    def open(self) -> None:
        self._t0 = time.monotonic()
        self._frame_id = 0
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def set_scene(self, scene: Scene) -> None:
        """测试注入新真值（下一帧生效）。"""
        self.scene = scene

    def read(self) -> Frame:
        if not self._opened:
            raise RuntimeError("SyntheticSource 未 open()；请用 with 语句")
        image = render(self.scene, self.width, self.height)
        frame = Frame(
            image=image,
            ts=time.monotonic(),
            frame_id=self._frame_id,
            source_fps=float(self.fps),
            scene=self.scene,
        )
        self._frame_id += 1
        return frame

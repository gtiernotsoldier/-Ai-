# -*- coding: utf-8 -*-
"""M1: 帧源契约与合成源能力。"""
import time

import numpy as np
import pytest

from core.screen_capture import SyntheticSource, ScrcpySource, Frame
from core.exceptions import DependencyMissing
from core.constants import FRAME_W, FRAME_H


def test_synthetic_source_contract():
    src = SyntheticSource(fps=60)
    with src:
        f1 = src.read()
        f2 = src.read()
    assert isinstance(f1, Frame)
    assert f1.image.shape == (FRAME_H, FRAME_W, 3)
    assert f1.image.dtype == np.uint8
    assert f2.frame_id == f1.frame_id + 1
    assert f2.ts >= f1.ts


def test_synthetic_not_opened_raises():
    src = SyntheticSource()
    with pytest.raises(RuntimeError):
        src.read()


def test_synthetic_60fps_capability():
    """合成源应远快于 60FPS（16.7ms/帧），为全链路留预算。"""
    src = SyntheticSource(fps=60)
    with src:
        t0 = time.perf_counter()
        for _ in range(120):
            src.read()
        per_frame = (time.perf_counter() - t0) / 120 * 1000
    assert per_frame < 16.7, f"合成帧生成 {per_frame:.2f}ms/帧, 超过 60FPS"


def test_scrcpy_missing_dependency_is_graceful():
    """沙箱无 scrcpy/av → 必须抛 DependencyMissing（而非裸异常/崩溃）。"""
    src = ScrcpySource()
    with pytest.raises(DependencyMissing):
        src.open()


def test_frame_never_persists():
    """原始画面即焚: Frame 不提供落盘方法。"""
    assert not hasattr(Frame, "save")
    assert not hasattr(Frame, "dump")

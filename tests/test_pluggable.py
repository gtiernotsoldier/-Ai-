# -*- coding: utf-8 -*-
"""可插拔依赖: 缺 ultralytics 时必须优雅降级, 且全链路仍可跑通。"""
import pytest

from core.detectors import MockFastDetector, YoloFastDetector
from core.exceptions import DependencyMissing
from core.keymap import Keymap
from core.perception import CVReader, Perception
from core.screen_capture import SyntheticSource
from core.state import StateBuilder
from core.synthetic import Scene


def test_yolo_missing_dependency_message():
    """沙箱无 ultralytics → DependencyMissing, 且报错信息含安装指引。"""
    pytest.importorskip_checked = None
    try:
        import ultralytics  # noqa: F401
        pytest.skip("本环境已装 ultralytics, 无法测降级路径")
    except ImportError:
        pass
    with pytest.raises(DependencyMissing) as ei:
        YoloFastDetector()
    assert "pip install ultralytics" in str(ei.value)


def test_full_chain_runs_without_heavy_deps():
    """零重依赖环境下: 捕获→感知→状态 全链路可用（轻依赖可插拔）。"""
    km = Keymap.load()
    perc = Perception(MockFastDetector(), CVReader(km), slow_every=5)
    builder = StateBuilder()
    src = SyntheticSource(scene=Scene(hp_self=0.9, cds={"ult": 0.5}))
    with src:
        frame = src.read()
        v = builder.build(perc.update(frame), frame.frame_id, frame.ts)
    assert v.shape == (64,) and v.dtype.name == "float32"

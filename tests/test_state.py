# -*- coding: utf-8 -*-
"""64 维状态向量: 维度/布局/取值范围/无NaN。"""
import math

import numpy as np

from core.state import StateBuilder, DEFAULT_LAYOUT, STATE_DIM
from core.perception import PerceptionResult
from core.synthetic import Scene
from core.screen_capture import SyntheticSource
from core.detectors import MockFastDetector
from core.keymap import Keymap
from core.perception import CVReader, Perception


def test_layout_covers_64_dims_exactly():
    described = set()
    for key, fields in DEFAULT_LAYOUT.describe().items():
        name = key.split("[")[0]
        start, end = map(int, key.split("[")[1].rstrip("]").split(":"))
        assert end - start == len(fields)
        described.update(range(start, end))
    assert described == set(range(STATE_DIM))


def test_build_shape_dtype():
    km = Keymap.load()
    src = SyntheticSource(scene=Scene(hp_self=0.8, cds={"skill1": 0.3}))
    perc = Perception(MockFastDetector(), CVReader(km), slow_every=5)
    builder = StateBuilder()
    with src:
        for i in range(6):
            frame = src.read()
            res = perc.update(frame)
    v = builder.build(res, frame.frame_id, frame.ts)
    assert v.shape == (STATE_DIM,)
    assert v.dtype == np.float32
    assert np.all(np.isfinite(v)), "状态向量必须无 NaN/Inf"
    assert np.all(v >= 0.0) and np.all(v <= 1.0), "全部取值须归一化 [0,1]"


def test_position_encoding():
    r = PerceptionResult(frame_id=0, ts=0.0,
                         self_pos=(640, 360), enemy_pos=(320, 180))
    v = StateBuilder().build(r)
    assert v[0] == 640 / 1280
    assert v[1] == 360 / 720
    assert v[8] == 320 / 1280
    # relative.dist: hypot(320,180)=367 -> /1500
    assert v[16] == pytest_approx(math.hypot(320, 180) / 1500)


def pytest_approx(x, abs=1e-5):
    import pytest
    return pytest.approx(x, abs=abs)


def test_slow_cadence_12hz():
    """慢变量严格 5 帧一读: 第0帧读数非零, 第1帧沿用缓存, 第5帧刷新。"""
    km = Keymap.load()
    src = SyntheticSource(scene=Scene(hp_self=0.5))
    perc = Perception(MockFastDetector(), CVReader(km), slow_every=5)
    with src:
        r0 = perc.update(src.read())
        assert r0.hp_self > 0.4                      # 第0帧就读
        r1 = perc.update(src.read())
        assert r1.hp_self == r0.hp_self              # 缓存
        for _ in range(3):
            src.read()
        r5 = perc.update(src.read())
        assert r5.hp_self > 0.4                      # 第5帧重新读


def test_event_trigger_via_effect_detection():
    """看到替身特效 → 自动起表。"""
    km = Keymap.load()
    perc = Perception(MockFastDetector(), CVReader(km), slow_every=5)
    perc.register_event("enemy_sub", duration_s=40.0)
    scene = Scene(effects={"sub_flash_enemy"})
    src = SyntheticSource(scene=scene)
    with src:
        res = perc.update(src.read())
    assert res.event_remaining["enemy_sub"] > 39.0

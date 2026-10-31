# -*- coding: utf-8 -*-
"""M1-M3 集成: 全链路 p95 不得超感知侧预算（红线防御的自动化形态）。"""
import time

import numpy as np

from core.detectors import MockFastDetector
from core.keymap import Keymap
from core.perception import CVReader, Perception
from core.screen_capture import SyntheticSource
from core.state import StateBuilder
from core.synthetic import Scene
from core.constants import MEASURABLE_STAGE_BUDGET


def test_pipeline_p95_within_budget():
    km = Keymap.load()
    perc = Perception(MockFastDetector(), CVReader(km), slow_every=5)
    builder = StateBuilder()
    src = SyntheticSource(scene=Scene(hp_self=0.73, cds={"skill1": 0.4}))
    totals = []
    with src:
        for _ in range(150):
            t0 = time.perf_counter_ns()
            frame = src.read()
            res = perc.update(frame)
            v = builder.build(res, frame.frame_id, frame.ts)
            totals.append((time.perf_counter_ns() - t0) / 1e6)
            assert v.shape == (64,)
    p95 = float(np.percentile(totals, 95))
    budget = (MEASURABLE_STAGE_BUDGET["capture"]
              + MEASURABLE_STAGE_BUDGET["fast_detect"]
              + MEASURABLE_STAGE_BUDGET["state"])
    assert p95 <= budget, f"全链路 p95={p95:.2f}ms > 预算合计 {budget}ms"

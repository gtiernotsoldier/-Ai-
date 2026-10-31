# -*- coding: utf-8 -*-
"""M2/M3: CV 读数精度对账 —— 写真值 → 渲染 → 读数 → 误差判定。"""
import pytest

from core.keymap import Keymap, KeymapError
from core.perception import CVReader, EventTimer
from core.synthetic import Scene
from core.screen_capture import SyntheticSource
from core.detectors import MockFastDetector

KM = Keymap.load()
CV = CVReader(KM)


def _frame(scene: Scene):
    src = SyntheticSource(scene=scene)
    with src:
        return src.read()


@pytest.mark.parametrize("hp", [1.0, 0.73, 0.45, 0.12, 0.0])
def test_hp_ratio_accuracy(hp):
    img = _frame(Scene(hp_self=hp)).image
    read = CV.hp_ratio(img, "self")
    assert read == pytest.approx(hp, abs=0.03), f"真值 {hp}, 读数 {read:.3f}"


@pytest.mark.parametrize("cd", [0.0, 0.25, 0.6, 0.9])
def test_cd_fraction_accuracy(cd):
    scene = Scene(cds={"skill1": cd})
    img = _frame(scene).image
    read = CV.cd_remaining(img, "skill1")
    assert read == pytest.approx(cd, abs=0.06), f"真值 {cd}, 读数 {read:.3f}"


def test_energy_ratio():
    img = _frame(Scene(energy_self=0.66)).image
    assert CV.energy_ratio(img, "self") == pytest.approx(0.66, abs=0.04)


def test_sub_ready_flag():
    assert CV.sub_ready(_frame(Scene(sub_ready_self=True)).image) is True
    assert CV.sub_ready(_frame(Scene(sub_ready_self=False)).image) is False


def test_keymap_rejects_out_of_bounds():
    import copy, json
    bad = json.loads(json.dumps({
        "frame": {"width": 1280, "height": 720},
        "regions": {"hp_self": {"x": 1200, "y": 30, "w": 300, "h": 18}},
    }))
    with pytest.raises(KeymapError):
        Keymap(bad)


def test_event_timer_with_fake_clock():
    t = [100.0]
    timer = EventTimer(duration_s=40.0, clock=lambda: t[0])
    assert timer.remaining() == 0.0
    timer.start()
    assert timer.remaining() == pytest.approx(40.0)
    t[0] += 15.0
    assert timer.remaining() == pytest.approx(25.0)
    assert timer.running
    t[0] += 30.0
    assert timer.remaining() == 0.0 and not timer.running


def test_mock_detector_outputs_detections():
    scene = Scene(self_pos=(400, 500), enemy_pos=(800, 400))
    dets = MockFastDetector().detect(_frame(scene))
    by_cls = {d.cls: d for d in dets}
    assert by_cls["self"].x + 30 == pytest.approx(400, abs=2)
    assert by_cls["enemy"].y + 30 == pytest.approx(400, abs=2)

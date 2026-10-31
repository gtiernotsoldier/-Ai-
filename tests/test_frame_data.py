# -*- coding: utf-8 -*-
"""M4.6 单测 — 帧数据表 (frame_data): 加载/校验/gap 推算/取消窗口语义。"""
from __future__ import annotations

import pytest

from core.frame_data import FrameData, FrameDataError


class TestLoadProduction:
    """生产配置 config/frame_data.json 的结构契约。"""

    @pytest.fixture(scope="class")
    def fd(self):
        return FrameData.load()

    def test_moves_complete(self, fd):
        assert set(fd.move_names) == {
            "atk", "skill1", "skill2", "skill3", "ult", "secret", "dash", "sub"}

    def test_combos_present(self, fd):
        assert {"bait_atk", "punish_sub", "basic_press"} <= set(fd.combo_names)

    def test_frame_ms_from_fps(self, fd):
        assert fd.fps == 60
        assert abs(fd.frame_ms - 1000.0 / 60.0) < 1e-9

    def test_atk_fields(self, fd):
        mv = fd.move("atk")
        assert (mv.startup, mv.active, mv.recovery) == (4, 3, 9)
        assert mv.total == 16
        assert mv.cancel_from == 10
        assert "atk" in mv.cancel_into and "skill3" in mv.cancel_into

    def test_gap_cancel_vs_full(self, fd):
        # atk→atk: 可取消衔接 → cancel_from=10 帧
        assert fd.move("atk").gap_to(fd.move("atk")) == 10
        # atk→ult: ult 不在 atk.cancel_into → 打满 total=16 帧
        assert fd.move("atk").gap_to(fd.move("ult")) == 16
        # skill2→atk: atk 在 skill2.cancel_into → cancel_from=24 帧
        assert fd.move("skill2").gap_to(fd.move("atk")) == 24

    def test_gap_s_seconds(self, fd):
        assert abs(fd.gap_s("atk", "atk") - 10 / 60.0) < 1e-9
        assert abs(fd.gap_s("atk", "ult") - 16 / 60.0) < 1e-9

    def test_unknown_move_raises(self, fd):
        with pytest.raises(FrameDataError):
            fd.move("flying_thunder_god")


class TestValidation:
    """坏表早炸: 帧约束/引用完整性在构造期校验。"""

    def _base(self):
        return {"fps": 60,
                "actions": {"atk": {"startup": 4, "active": 3, "recovery": 9,
                                    "cancel_from": 10,
                                    "cancel_into": ["atk"]}},
                "combos": {}}

    def test_cancel_from_gt_total_raises(self):
        raw = self._base()
        raw["actions"]["atk"]["cancel_from"] = 17      # total=16
        with pytest.raises(FrameDataError):
            FrameData(raw)

    def test_cancel_from_lt_startup_raises(self):
        raw = self._base()
        raw["actions"]["atk"]["cancel_from"] = 2       # startup=4
        with pytest.raises(FrameDataError):
            FrameData(raw)

    def test_cancel_into_unknown_ref_raises(self):
        raw = self._base()
        raw["actions"]["atk"]["cancel_into"] = ["nope"]
        with pytest.raises(FrameDataError):
            FrameData(raw)

    def test_combo_unknown_ref_raises(self):
        raw = self._base()
        raw["combos"] = {"c": {"steps": ["ghost"]}}
        with pytest.raises(FrameDataError):
            FrameData(raw)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FrameDataError):
            FrameData.load(tmp_path / "absent.json")

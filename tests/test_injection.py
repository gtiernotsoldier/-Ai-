# -*- coding: utf-8 -*-
"""M4 单测 — 触控注入 (injection)。"""
from __future__ import annotations

import pytest

from core.exceptions import DependencyMissing
from core.injection import MockInjector, MinitouchInjector, TouchEvent


class TestTouchEvent:
    def test_valid_actions(self):
        for a in ("d", "m", "u"):
            TouchEvent(0, 100, 200, a)

    def test_invalid_action(self):
        with pytest.raises(ValueError):
            TouchEvent(0, 100, 200, "x")

    def test_frozen(self):
        ev = TouchEvent(0, 1, 2, "d")
        with pytest.raises(Exception):
            ev.x = 3


class TestMockInjector:
    def test_press_tracking(self):
        inj = MockInjector()
        inj.open()
        inj.inject([TouchEvent(0, 10, 20, "d")])
        assert inj.contacts_pressed() == {0}
        inj.inject([TouchEvent(0, 30, 40, "m")])
        assert inj.contacts_pressed() == {0}
        inj.inject([TouchEvent(0, 30, 40, "u")])
        assert inj.contacts_pressed() == set()

    def test_multi_contact_independent(self):
        inj = MockInjector()
        inj.open()
        inj.inject([TouchEvent(0, 10, 20, "d"), TouchEvent(1, 100, 200, "d")])
        assert inj.contacts_pressed() == {0, 1}
        inj.inject([TouchEvent(0, 10, 20, "u")])
        assert inj.contacts_pressed() == {1}

    def test_reset_releases_all(self):
        inj = MockInjector()
        inj.open()
        inj.inject([TouchEvent(0, 1, 1, "d"), TouchEvent(1, 2, 2, "d"),
                    TouchEvent(2, 3, 3, "d")])
        ms = inj.reset()
        assert ms >= 0.0
        assert inj.contacts_pressed() == set()
        resets = [e for e in inj.events if e.kind == "reset"]
        assert len(resets) == 3            # 3 个按压触点逐一补 up

    def test_close_releases_residual(self):
        inj = MockInjector()
        inj.open()
        inj.inject([TouchEvent(0, 1, 1, "d")])
        inj.close()
        assert inj.contacts_pressed() == set()

    def test_inject_returns_elapsed_ms(self):
        inj = MockInjector()
        inj.open()
        ms = inj.inject([TouchEvent(0, 1, 1, "d")])
        assert isinstance(ms, float) and ms >= 0.0

    def test_events_snapshot(self):
        inj = MockInjector()
        inj.open()
        inj.inject([TouchEvent(0, 1, 1, "d")])
        snap = inj.events
        inj.inject([TouchEvent(0, 1, 1, "u")])
        assert len(snap) == 1 and len(inj.events) == 2   # 快照不被后续污染


class TestMinitouchInjector:
    def test_open_without_adb_raises_graceful(self, monkeypatch):
        """沙箱无 adb → DependencyMissing（优雅降级, 不崩溃）。"""
        monkeypatch.setattr("core.injection.shutil.which", lambda _: None)
        inj = MinitouchInjector()
        with pytest.raises(DependencyMissing) as ei:
            inj.open()
        assert ei.value.module == "adb"

    def test_scale_mapping(self):
        inj = MinitouchInjector()
        inj.max_x, inj.max_y = 2560, 1440            # 模拟真机 banner
        x, y = inj._scale(TouchEvent(0, 640, 360, "d"))
        assert (x, y) == (1280, 720)
        x, y = inj._scale(TouchEvent(0, 1280, 720, "m"))
        assert (x, y) == (2560, 1440)

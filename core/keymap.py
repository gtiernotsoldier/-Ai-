# -*- coding: utf-8 -*-
"""keymap.json 加载与校验（M2: 标定产物是全感知的地基）。

keymap 描述每个 UI 区域在 1280x720 画面上的像素位置。
换分辨率/换模拟器皮肤时只需重新标定本文件，感知代码零改动。
"""
from __future__ import annotations

import json
from pathlib import Path

from core.constants import FRAME_W, FRAME_H

DEFAULT_KEYMAP_PATH = Path(__file__).resolve().parent.parent / "config" / "keymap.json"


class KeymapError(ValueError):
    pass


class Keymap:
    def __init__(self, data: dict):
        self.frame = data.get("frame", {})
        self.regions: dict[str, dict] = data.get("regions", {})
        if self.frame.get("width") != FRAME_W or self.frame.get("height") != FRAME_H:
            raise KeymapError(
                f"keymap 分辨率必须为 {FRAME_W}x{FRAME_H}（与 scrcpy 推流分辨率一致），"
                f"实际 {self.frame}"
            )
        self._validate()

    def _validate(self) -> None:
        for name, r in self.regions.items():
            if "cx" in r:   # 圆形区域（按钮）
                if not (0 <= r["cx"] - r["r"] and r["cx"] + r["r"] <= FRAME_W
                        and 0 <= r["cy"] - r["r"] and r["cy"] + r["r"] <= FRAME_H):
                    raise KeymapError(f"区域 {name} 超出画面边界: {r}")
            else:           # 矩形区域（血条/能量条）
                if not (r["x"] >= 0 and r["y"] >= 0
                        and r["x"] + r["w"] <= FRAME_W
                        and r["y"] + r["h"] <= FRAME_H):
                    raise KeymapError(f"区域 {name} 超出画面边界: {r}")

    # ── 便捷取值 ──
    def rect(self, name: str) -> tuple[int, int, int, int]:
        r = self.regions[name]
        return r["x"], r["y"], r["w"], r["h"]

    def circle(self, name: str) -> tuple[int, int, int]:
        r = self.regions[name]
        return r["cx"], r["cy"], r["r"]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Keymap":
        p = Path(path) if path else DEFAULT_KEYMAP_PATH
        if not p.exists():
            raise KeymapError(f"keymap 文件不存在: {p}（先运行 tools/calibrate_keymap.py）")
        with open(p, encoding="utf-8") as f:
            return cls(json.load(f))

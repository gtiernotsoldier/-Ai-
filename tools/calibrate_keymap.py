# -*- coding: utf-8 -*-
"""M2 — keymap 标定工具。

用法:
  # 1) 对一张截图/合成帧画出所有标定区域 → 标定图
  python -m tools.calibrate_keymap --frame frame.png --keymap config/keymap.json -o calibrated.png

  # 2) 校验 keymap 合法性（区域越界/分辨率不匹配直接报错）
  python -m tools.calibrate_keymap --check

真机标定流程: 截一帧对局画面 → 用画图工具量出各区域像素坐标 → 填 keymap.json
→ 跑 --check 与 --frame 验证 → 感知层即可工作。
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from core.keymap import Keymap, KeymapError, DEFAULT_KEYMAP_PATH

_OVERLAY_COLORS = {
    "hp": (0, 255, 0), "energy": (255, 200, 0),
    "btn": (0, 160, 255), "joystick": (255, 0, 200),
}


def _color(name: str) -> tuple[int, int, int]:
    for k, c in _OVERLAY_COLORS.items():
        if name.startswith(k):
            return c
    return (255, 255, 255)


def annotate(frame_img: np.ndarray, km: Keymap) -> np.ndarray:
    out = frame_img.copy()
    for name, r in km.regions.items():
        c = _color(name)
        if "cx" in r:
            cv2.circle(out, (r["cx"], r["cy"]), r["r"], c, 2)
            cv2.drawMarker(out, (r["cx"], r["cy"]), c, cv2.MARKER_CROSS, 12, 2)
        else:
            cv2.rectangle(out, (r["x"], r["y"]),
                          (r["x"] + r["w"], r["y"] + r["h"]), c, 2)
        label_xy = (r.get("x", r["cx"] - r["r"]), r.get("y", r["cy"] - r["r"]) - 6)
        cv2.putText(out, name, label_xy, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, c, 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Shinobi_God keymap 标定工具 (M2)")
    ap.add_argument("--keymap", default=str(DEFAULT_KEYMAP_PATH))
    ap.add_argument("--frame", help="待标注的画面 png/jpg")
    ap.add_argument("-o", "--output", default="calibrated.png")
    ap.add_argument("--check", action="store_true", help="仅校验 keymap 合法性")
    args = ap.parse_args()

    try:
        km = Keymap.load(args.keymap)
    except KeymapError as e:
        print(f"[FAIL] {e}")
        return 2
    print(f"[OK] keymap 合法: {args.keymap}（{len(km.regions)} 个区域, "
          f"{km.frame['width']}x{km.frame['height']}）")

    if args.check and not args.frame:
        return 0
    if args.frame:
        img = cv2.imread(args.frame)
        if img is None:
            print(f"[FAIL] 无法读取画面: {args.frame}")
            return 2
        cv2.imwrite(args.output, annotate(img, km))
        print(f"[OK] 标定图已写出: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

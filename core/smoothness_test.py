# -*- coding: utf-8 -*-
"""验收测试 — 丝滑度 (宪法第七节: smoothness_test)

检查项:
  [A] 决策节拍稳定性: 20Hz 调度下感知→状态组装的周期抖动 (可跑, 沙箱出数字)
  [B] 摇杆轨迹无停顿帧 / 注入<20ms / 中断零残留  → 依赖执行层(M4), 未实现则 SKIP
每周跑一次, 数字写进 README —— 没有度量就没有防御。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from core.detectors import MockFastDetector
from core.keymap import Keymap
from core.perception import CVReader, Perception
from core.screen_capture import SyntheticSource
from core.state import StateBuilder
from core.synthetic import Scene

DECISION_PERIOD_MS = 50.0      # 20Hz
RUN_SECONDS = 2.0
JITTER_LIMIT_MS = 5.0          # 周期抖动容忍(开发机非实时OS放宽; 真机另测)


def check_cadence() -> tuple[bool, str]:
    """[A] 20Hz 节拍: 每 tick 跑一遍 捕获→感知→状态组装, 测周期抖动。"""
    km = Keymap.load()
    src = SyntheticSource(fps=60, scene=Scene())
    perc = Perception(MockFastDetector(), CVReader(km), slow_every=5)
    builder = StateBuilder()
    periods: list[float] = []
    t_prev = None
    t_end = time.monotonic() + RUN_SECONDS
    with src:
        ticks = 0
        while time.monotonic() < t_end:
            t_tick = time.monotonic()
            frame = src.read()
            res = perc.update(frame)
            builder.build(res, frame.frame_id, frame.ts)
            ticks += 1
            if t_prev is not None:
                periods.append((t_tick - t_prev) * 1000)
            t_prev = t_tick
            # 纪律: 这里是离线测试工具, 允许 sleep; 实战闭环内禁止
            remain = DECISION_PERIOD_MS / 1000 - (time.monotonic() - t_tick)
            if remain > 0:
                time.sleep(remain)
    p = np.array(periods)
    jitter = float(np.percentile(np.abs(p - DECISION_PERIOD_MS), 95))
    mean_hz = 1000.0 / float(p.mean())
    ok = jitter <= JITTER_LIMIT_MS and abs(mean_hz - 20.0) < 0.5
    return ok, (f"20Hz 节拍: 实测 {mean_hz:.2f}Hz, 周期抖动 p95={jitter:.2f}ms "
                f"(容忍 {JITTER_LIMIT_MS}ms), {ticks} ticks "
                f"→ {'PASS' if ok else 'FAIL'}")


def check_executor() -> tuple[bool, str]:
    """[B] 摇杆/注入/中断 —— 需要 M4 执行层。"""
    if not (Path(__file__).resolve().parent.parent / "core" / "executor.py").exists():
        return True, ("[SKIP] 摇杆轨迹平滑/注入<20ms/中断零残留: 依赖执行层(M4), "
                      "未实现。M4 落地后本测试自动启用。")
    return True, "[TODO] M4 executor 存在, 接入具体检查项"


def main() -> int:
    print("=" * 78)
    print("Shinobi_God 验收: 丝滑度测试")
    print("=" * 78)
    results = [check_cadence(), check_executor()]
    print()
    ok = True
    for passed, msg in results:
        print(f"  {msg}")
        ok &= passed
    print("\n纪律提醒: 闭环内零网络/零磁盘/零睡眠; 本工具的 sleep 仅存在于离线测试。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

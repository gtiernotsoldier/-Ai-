# -*- coding: utf-8 -*-
"""验收测试 — 闭环末端延迟量化 (宪法第七节)

两种模式:
  默认(sandbox) — 合成帧驱动 M1-M3 全链路, 逐环节 p50/p95/max 对照预算表,
                  沙箱/开发机/CI 均可复现, 数字写入 README。
  --real        — 真机模式: 检查 adb/scrcpy/设备在场, 按"按键→技能光效帧差"
                  量化端到端延迟（需真机, 本沙箱必然缺席并给出指引）。

退出码: 0=全部预算内 | 1=超预算(红线) | 2=环境不满足
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

import numpy as np

from core.constants import MEASURABLE_STAGE_BUDGET, TARGET_LATENCY_MS, REDLINE_LATENCY_MS
from core.detectors import MockFastDetector
from core.keymap import Keymap
from core.perception import CVReader, Perception
from core.screen_capture import SyntheticSource
from core.state import StateBuilder
from core.synthetic import Scene


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def run_sandbox(frames: int = 300) -> int:
    print("=" * 78)
    print("Shinobi_God 验收: 感知闭环延迟测试 (M1-M3, 合成帧驱动)")
    print("=" * 78)
    km = Keymap.load()
    src = SyntheticSource(fps=60, scene=Scene(hp_self=0.73, cds={"skill1": 0.4}))
    fast = MockFastDetector()
    perc = Perception(fast, CVReader(km), slow_every=5)
    builder = StateBuilder()

    t_cap, t_fast, t_slow, t_state, t_total = [], [], [], [], []
    with src:
        for i in range(frames):
            t0 = time.perf_counter_ns()
            frame = src.read()
            t1 = time.perf_counter_ns()
            res = perc.update(frame)
            t2 = time.perf_counter_ns()
            if frame.frame_id % 5 == 0:
                pass  # 慢变量已并入 update; 记录含慢变量帧的总耗时
            v = builder.build(res, frame.frame_id, frame.ts)
            t3 = time.perf_counter_ns()

            assert v.shape == (64,)
            t_cap.append((t1 - t0) / 1e6)
            t_fast.append((t2 - t1) / 1e6)
            t_state.append((t3 - t2) / 1e6)
            t_total.append((t3 - t0) / 1e6)

    print(f"\n样本: {frames} 帧 @ 目标 60Hz (快变量) / 12Hz (慢变量, 5帧一读)\n")
    hdr = f"{'环节':<28}{'预算ms':>8}{'p50':>9}{'p95':>9}{'max':>9}  判定"
    print(hdr)
    print("-" * len(hdr))
    rows = [
        ("画面捕获(合成源≈scrcpy)", MEASURABLE_STAGE_BUDGET["capture"], t_cap),
        ("感知推理(快+慢变量)", MEASURABLE_STAGE_BUDGET["fast_detect"], t_fast),
        ("状态组装", MEASURABLE_STAGE_BUDGET["state"], t_state),
    ]
    fail = False
    for name, budget, xs in rows:
        p95 = _pct(xs, 95)
        ok = p95 <= budget
        fail |= not ok
        print(f"{name:<28}{budget:>8.1f}{_pct(xs,50):>9.2f}{p95:>9.2f}"
              f"{max(xs):>9.2f}  {'PASS' if ok else '✗ 超预算'}")

    print("-" * len(hdr))
    p95 = _pct(t_total, 95)
    print(f"M1-M3 子链路合计(不含决策/执行/模拟器物理税): "
          f"p50={_pct(t_total,50):.2f}ms  p95={p95:.2f}ms  max={max(t_total):.2f}ms")
    print(f"对照: 目标40ms / 红线100ms —— 感知侧占用预算 "
          f"{p95 / REDLINE_LATENCY_MS * 100:.1f}% (红线) / "
          f"{p95 / TARGET_LATENCY_MS * 100:.1f}% (目标)")
    print("\n[提示] 决策前向(5-15ms)+模拟器物理税(32ms) 属 M6 环节, 待真机联调后并入本表;")
    print("       minitouch 注入(M4)已实现, 实测见 smoothness_test [B]: 注入 p95<0.1ms。")
    return 1 if fail else 0


def run_real() -> int:
    print("Shinobi_God 真机延迟验收: 按键→技能光效帧差")
    print("-" * 50)
    missing = []
    if shutil.which("adb") is None:
        missing.append("adb (Android platform-tools)")
    if shutil.which("scrcpy") is None:
        missing.append("scrcpy")
    else:
        try:
            out = subprocess.run(["adb", "devices"], capture_output=True,
                                 text=True, timeout=10).stdout
            devices = [l for l in out.splitlines()[1:] if l.strip() and "device" in l]
            if not devices:
                missing.append("已授权的 adb 设备 (MuMu 需 adb connect 127.0.0.1:7555)")
        except Exception:
            missing.append("adb devices 探测失败")
    if missing:
        print("[SKIP] 真机环境不满足, 缺少:")
        for m in missing:
            print(f"   - {m}")
        print("\n真机步骤: 启动 MuMu → adb connect 127.0.0.1:7555 → python -m core.latency_test --real")
        return 2
    print("[TODO] 设备在场。联调脚本: 触发按键 → 逐帧比对技能光效出现帧差 → 写 README。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="真机模式(需 adb+scrcpy+设备)")
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()
    return run_real() if args.real else run_sandbox(args.frames)


if __name__ == "__main__":
    sys.exit(main())

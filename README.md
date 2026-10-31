> 本文件为项目自述。完整架构见对话中 v1.0 文档；此处仅放运行与验收信息。

# Shinobi_God

火影手游纯视觉强化学习智能体 —— 单机规模，绝悟方案缩量复刻。

## 宪法红线
1. 纯视觉输入（不读内存/不注入进程）
2. 感知→决策→执行→生效 ≤100ms（目标 40ms）
3. 闭环内零网络/零磁盘/零睡眠；慢事走 ZMQ 后台
4. 拟人化输出永远开启（120ms 反应延迟 + 1% 失误）

## 当前进度（路线图）

| 里程碑 | 状态 | 说明 |
|---|---|---|
| M1 screen_capture | ✅ 完成 | FrameSource 抽象: ScrcpySource(真机,缺依赖优雅降级) + SyntheticSource(沙箱全功能) |
| M2 keymap 标定 | ✅ 完成 | config/keymap.json + tools/calibrate_keymap.py（--check/--frame 标注） |
| M3 感知层 60Hz | ✅ 完成 | 快变量(可插拔YOLO/Mock) + 慢变量(纯OpenCV血条/能量/CD) + 事件计时器 → 64维状态向量 |
| M4 宏引擎+minitouch | ✅ 完成 | injection(Minitouch/Mock) + macro(Tap/Swipe/Joystick 20Hz/贝塞尔) + executor(可中断宏协议+拟人化) |
| M5 录像+BC | ⬜ | CSPL 阶段① |
| M6 PPO 自对弈 | ⬜ | CSPL 阶段③ |
| M7 蒸馏合并 | ⬜ | CSPL 阶段② |

## 快速开始

```bash
pip install -r requirements.txt          # 轻依赖: numpy/opencv/pytest
python -m core.latency_test              # 验收: 感知闭环延迟（数字见下）
python -m core.smoothness_test           # 验收: 丝滑度
python -m tools.calibrate_keymap --check # M2 keymap 校验
pytest -q                                # 单测
```

可选重依赖（真机/真推理时安装）:
```bash
pip install ultralytics   # YOLOv8n 快变量
pip install onnxruntime   # 决策层 FP16 前向 (M4+)
pip install av            # scrcpy 裸流解码
```

## 验收数字（沙箱实测, Python 3.12 / numpy 2.1 / OpenCV 4.13）

`python -m core.latency_test`（300 帧, 合成帧驱动 M1-M3 全链路）:

| 环节 | 预算 | p50 | p95 | max | 判定 |
|---|---|---|---|---|---|
| 画面捕获(合成源≈scrcpy) | 10ms | 6.59 | 8.93 | 32.62 | PASS |
| 感知推理(快+慢变量) | 15ms | 0.02 | 0.20 | 0.53 | PASS |
| 状态组装 | 1ms | 0.14 | 0.21 | 0.52 | PASS |

M1-M3 子链路合计 p95=9.22ms —— 占红线 100ms 的 9.2%、目标 40ms 的 23.1%，
为决策前向(5-15ms)与模拟器物理税(32ms)留足余量。

`python -m core.smoothness_test`:
- [A] 20Hz 节拍实测 19.97Hz, 周期抖动 p95=0.09ms → PASS
- [B] 摇杆流 22 事件/秒, 间隔 p95=50.01ms max=50.01ms (容忍 75ms) → PASS；
  注入 p95=0.018ms (<20ms) → PASS；中断响应 0.20ms, 残留触点 0 → PASS

`pytest -q`: 66 passed（含 CV 读数精度对账: 血条 ±0.03 / CD ±0.014 / 能量 ±0.04;
M4 中断协议: 零残留 + Event.wait 即时唤醒）。

## 目录结构

```
core/          constants(宪法) | screen_capture(M1) | keymap+perception(M2/M3)
               state(64维) | detectors(可插拔) | injection+macro+executor(M4)
               latency_test | smoothness_test
config/        keymap.json（标定产物）
tools/         calibrate_keymap.py | synthetic.py
tests/         pytest 单测
```

## 真机联调（M4 执行层）

```bash
# 前置: adb + 设备端 minitouch（abrutus/minitouch）
adb connect 127.0.0.1:7555      # MuMu 模拟器
python -m core.latency_test --real
```
注入器接口与 Mock 完全同构: `MinitouchInjector(serial, port=11111)` 换入 Executor 即可,
分辨率映射由 minitouch banner 自动完成(1280x720 → 真机分辨率)。

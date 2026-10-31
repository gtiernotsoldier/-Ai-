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
| M4.5 决策接口+主循环 | ✅ 完成 | policy(Action空间/Rule/Onnx/看门狗) + orchestrator(20Hz 闭环: 捕获→感知→状态→决策→执行, 沙箱自走) |
| 打磨 P0/P1/P2 | ✅ 完成 | P0-3 三键(atk/secret/dash) + P1 死维度回填(24→36)与张冠李戴修复 + P2 敌踪记忆/CD量纲; 见《打磨清单》 |
| M4.6 帧数据表+Sequencer | ✅ 完成 | ⑥帧数据表+动作时序状态机 ⑦bait→sub_flash→punish 事件分支 ⑩20Hz策略/60Hz帧级分层 |
| M4.7 检测器扩类+博弈层 | ✅ 完成 | ⑧projectile/hitbox_warn/wall+威胁TTC+墙体 ⑨TradeScorer+IntentPolicy 决策梯(躲避/逃墙角/逼墙/bait-punish)(197 tests) |
| M5 录像+BC | ⬜ | CSPL 阶段① |
| M6 PPO 自对弈 | ⬜ | CSPL 阶段③ |
| M7 蒸馏合并 | ⬜ | CSPL 阶段② |

## 快速开始

```bash
pip install -r requirements.txt          # 轻依赖: numpy/opencv/pytest
python -m core.latency_test              # 验收: 感知闭环延迟（数字见下）
python -m core.smoothness_test           # 验收: 丝滑度
python -m core.orchestrator              # 演示: 20Hz 全链路自走 + 闭环延迟报告
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

`python -m core.orchestrator`（20Hz 全链路自走 3s）:

| 环节 | 预算 | p50 | p95 | max | 判定 |
|---|---|---|---|---|---|
| 画面捕获 | 10ms | 6.75 | 7.22 | 7.66 | PASS |
| 感知推理 | 15ms | 0.04 | 0.36 | 0.43 | PASS |
| 状态组装 | 1ms | 0.23 | 0.26 | 0.29 | PASS |
| 决策(看门狗20ms) | 15ms | 0.01 | 0.02 | 0.02 | PASS |
| 动作下发 | 1ms | 0.02 | 0.03 | 0.23 | PASS |
| 闭环合计(捕获起) | 42ms | 7.10 | 7.68 | 8.03 | PASS |

实测节拍 19.96Hz, 看门狗 0 触发, 残留触点 0 —— 闭环 p95 占红线 7.7%/目标 19.2%
(不含模拟器渲染 16ms + 响应 16ms 物理税)。决策层换 OnnxPolicy 时同表重跑。

`python -m core.smoothness_test`:
- [A] 20Hz 节拍实测 19.97Hz, 周期抖动 p95=0.09ms → PASS
- [B] 摇杆流 22 事件/秒, 间隔 p95=50.01ms max=50.01ms (容忍 75ms) → PASS；
  注入 p95=0.018ms (<20ms) → PASS；中断响应 0.20ms, 残留触点 0 → PASS

`pytest -q`: 197 passed（含 CV 读数精度对账: 血条 ±0.03 / CD ±0.014 / 能量 ±0.04;
M4 中断协议: 零残留 + Event.wait 即时唤醒; M4.5 看门狗沿用上帧 + 零残留收尾;
**走位持久流回归**: 对手持续移动时每 tick 方向都在变, 摇杆仍只按下 1 次且 20Hz 不断流;
**打磨 P 级回归**: 三键执行/张冠李戴/敌踪衰减/CD量纲/buttons 反馈闭环;
**M4.6 Sequencer**: 连段一次下发不被逐tick重建/取消窗口替换/ bait→替身闪→punish 端到端;
**M4.7 博弈层**: 弹道 TTC(侧向脱靶不误报)/预警圈/墙距/flank/决策梯全梯级/Sequencer 接线, 见下表）。

### 打磨清单 P 级完成记录

| 项 | 内容 | 状态 |
|---|---|---|
| P0-1/P0-2/P0-5 | 走位断流/拟人分层/触点分路（见上节历史坑） | ✅ 前次已修 |
| P0-3 | atk/secret/dash 三键: ACTION_BUTTONS 扩至 9-16 + keymap 三区域（占位需真机标定） | ✅ |
| P0-4 | hold_ms/strength/CHARGE_HOLD_MS（前次已完成, 本次核验） | ✅ |
| P1 | enemy_kin.sub_ready 张冠李戴修复; 死维度回填 24→36/64（buttons 切片动作反馈、
closing_speed/approach/threat 语义修复、resources 敌我双侧、meta r6-r9 计时器+置信度） | ✅ |
| P2#4 | 敌人失踪保留最后已知位置 + exp(-age/τ) 置信度衰减 | ✅ |
| P2#5 | resources.ult_cd_self(秒/60) 与 skill_cds.ult(按总CD) 量纲分离 | ✅ |
| ① | 事件计时器生产路径注册(enemy_sub/self_sub/enemy_ult)+特效起表+敌方替身就绪接回状态 | ✅ |
| ⑤ | buttons 切片 6 维由上一帧实际下发动作回填（Orchestrator._buttons_feedback） | ✅ |

### M4.6 帧数据表 + Sequencer（施工⑥⑦⑩）

| 项 | 内容 | 状态 |
|---|---|---|
| ⑥帧数据表 | `config/frame_data.json`(startup/active/recovery/cancel_from/cancel_into + 连段表) + `core/frame_data.py`(加载/校验/取消衔接 gap 推算) | ✅ |
| ⑥动作时序状态机 | `core/action_state.py` ActionTracker: 回答「现在是这招第几帧、取消窗口开了没」 | ✅ |
| ⑩架构分层 | `core/sequencer.py`: 20Hz 策略出意图 → Sequencer 按帧表一次编译一次下发;
取消窗口内替换=取消后摇, 窗口未开拒绝(20Hz 刷屏拆不散连段); 走位仍由摇杆持久流承载 | ✅ |
| ⑦卡替身 | bait 序列声明分支(事件,替代连段); 感知事件边沿 events_fired → Orchestrator
直接转 Sequencer.on_events → 帧级切 punish(不经 20Hz 回合); 端到端测试验证 | ✅ |
| 策略层接口 | 策略可选实现 bind_sequencer()(意图式), 未实现则按钮意图照旧单步下发(完全兼容) | ✅ |

### M4.7 检测器扩类 + 博弈策略层（施工⑧⑨）

| 项 | 内容 | 状态 |
|---|---|---|
| ⑧检测器扩类 | YOLO 类别表扩至 13: projectile/hitbox_warn/wall + 6 个自身状态特效类;
Detection 携带 facing; MockFastDetector 同构产出（合成 Scene 新增 Projectile/HitboxWarn/walls/facing_enemy 真值） | ✅ |
| ⑧威胁模型 | `core/threat.py` ThreatTracker: 弹道跨帧关联 → 最近逼近法算 TTC(含脱靶判定,
侧向飞过不误报); hitbox_warn 圈内即威胁; WallModel: 墙框优先/边界兜底的墙距+最近墙点 | ✅ |
| ⑧状态回填 | threat_ttc/threat_dir(meta 10/11)、wall_dist_self/enemy(2/3)、effects 8 维全回填、
flank(绕后)、engage(敌逼近)、敌方朝向、dash_cd(按过 dash 起表); meta r6-r11 语义化改名 | ✅ |
| ⑨TradeScorer | 换血评分: 期望伤害−被反打风险(敌替身就绪×1.6/大招威胁×2.0)+血量优势项,
规则化可解释; DAMAGE_EST 占位待标定 | ✅ |
| ⑨IntentPolicy | 博弈决策梯: 被控停手→威胁侧移+强制dash→残血替身→逃墙角→逼墙压制(替身不在
直接 punish)→bait(带 enemy_sub 事件分支)→trade 进攻→拉开/接近; 只读 64 维状态,
bind_sequencer 提交序列意图, 未绑定退化单帧 Action | ✅ |
| 执行反馈 | state.meta.action_progress(当前招式帧进度, Sequencer.progress)+watchdog 进状态;
自己按 dash → self_dash 计时器(dash_cd 来源) | ✅ |

【占位值提醒】`CD_TOTAL_S` / `EVENT_CD_S` / `CHARGE_HOLD_MS` / `HITBOX_WARN_TTC_S` /
`DAMAGE_EST` / `frame_data.json` 全部帧数与 btn_atk/btn_secret/btn_dash 坐标均为占位,
需真机实测标定后覆盖（标定前勿作训练标签依据）。

【状态向量契约 v2】meta 切片 r6-r11 已语义化改名(enemy_sub_s/self_sub_s/enemy_ult_s/
enemy_pos_conf/threat_ttc/threat_dir), slow_age/fast_age→wall_dist_self/enemy,
decision_gauge→action_progress —— 决策/训练接口以 `StateLayout.describe()` 为准
(尚无 onnx 导出, 训练前以此版定稿)。

### 走位下发机制（摇杆持久流）

走位与按钮**分路下发**，这是 P0-1 修复后的形态：

- **走位 → `JoystickStream` 持久流**：整局只按下一次（`CONTACT_JOYSTICK`），
  之后每 tick 只是在线更新方向，转向**不释放触点、不重建宏、不加反应延迟**；
  方向变化时允许提前下发（下限 20ms），把最坏转向延迟从 50ms 压到 20ms。
- **按钮 → 一次性宏**：走 `CONTACT_BUTTON`，与摇杆并行不互顶 →「边走边放技能」成立。
  宏收尾只 `release()` 自己的触点，**不再用全局 `reset()`**（全局 reset 会把摇杆一起松掉）。
- **拟人化分层**：`HUMANIZE_REACTION_MS=120ms` 只施加在「反应型」动作上
  （`Macro(humanize=True)`，默认值）；「序列型」宏（连段内部/续流）用
  `Macro(humanize=False)` 走零延迟精确排程。1% 指令级失误扰动不受此开关影响，始终开启。

> 历史坑：旧实现每 tick 用浮点相等判「同向续流」，方向差一个尾数就中断重建摇杆宏，
> 而重建的宏首事件前有 120ms 拟人延迟 > 50ms 节拍 → 宏永远活不到按下事件。
> 实测方向每 tick 变 0.05° 时，2 秒内按下事件数 = 0（智能体完全不动）；
> 而静止 Scene 的 demo 因为动作逐帧完全相同恰好走了续流快路径，把 bug 盖住了。
> 现在 `LoopReport.stick_starts` 是这条的回归哨兵：连续走位期间应恒为 1。

## 目录结构

```
core/          constants(宪法) | screen_capture(M1) | keymap+perception(M2/M3)
               state(64维) | detectors(可插拔) | injection+macro+executor(M4)
               policy+orchestrator(M4.5) | frame_data+action_state+sequencer(M4.6)
               threat+strategy(M4.7) | latency_test | smoothness_test
config/        keymap.json（标定产物, 含 atk/secret/dash 占位区域） | frame_data.json
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

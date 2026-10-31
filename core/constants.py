# -*- coding: utf-8 -*-
"""Shinobi_God 宪法常量（v1.0 终版）

本模块是全项目的单一事实来源（Single Source of Truth）。
任何模块不得私自定义与延迟/频率/维度相关的魔法数字。
"""
from __future__ import annotations

# ───────────────────────── 红线（不可违反） ─────────────────────────
REDLINE_LATENCY_MS = 100      # 感知→决策→执行→生效 红线
TARGET_LATENCY_MS = 40        # 实战闭环目标
STATE_DIM = 64                # 状态向量维度

# 闭环纪律: 零网络 / 零磁盘 / 零睡眠（慢事走 ZMQ 后台进程）
CLOSED_LOOP_FORBIDDEN = ("network", "disk_io", "sleep")

# 拟人化输出（永远开启）
HUMANIZE_REACTION_MS = 120    # 策略级反应延迟（仅「反应型」动作; 连段/续流内部事件不加）
HUMANIZE_MISS_RATE = 0.01     # 指令级扰动/失误率
HUMANIZE_MISS_PX = 12.0       # 失误扰动最大像素偏移

# 普攻点按 / 长按(蓄力) 的分界。⚠️ 占位值, 需从实战录像标定后覆盖:
# 低于此值视为点按, 高于视为长按。标定前不要拿它做训练数据的标签依据。
CHARGE_HOLD_MS = 200.0

# ───────────────────────── M4 执行层 ─────────────────────────
EXEC_INJECT_LIMIT_MS = 20.0   # 注入/中断响应上限（丝滑度[B]看门狗口径）
GAP_TOLERANCE_X = 1.5         # 摇杆流相邻事件间隔容忍倍数 (50ms*1.5=75ms)

# ───────────────────────── M4.5 决策接口 + 主循环 ─────────────────────────
# 闭环本地可测子集预算: 捕获+感知+组装+决策+下发（模拟器渲染/响应两段物理税除外）
LOOP_BUDGET_MS = 42.0
# 离散动作表（OnnxPolicy argmax → 动作）: 0-8 摇杆8向+静止, 9-16 按钮
# P0-3: atk(普攻)/secret(秘卷)/dash(闪避) —— 此前三键不存在, 无法执行
ACTION_BUTTONS: tuple[str, ...] = ("skill1", "skill2", "skill3", "ult", "sub",
                                   "atk", "secret", "dash")

# ───────────── 技能总 CD 表（秒）【占位值, 需按角色实测标定后覆盖】 ─────────────
# 用途: CVReader 扇形占比(剩余/总CD) × 总CD = 剩余秒数 —— 资源置换判断需要时间量纲
CD_TOTAL_S: dict[str, float] = {
    "skill1": 12.0, "skill2": 16.0, "skill3": 20.0,
    "ult": 45.0, "sub": 40.0, "secret": 30.0,
}

# 事件计时器时长（秒）【占位值, 需按角色实测标定】—— 打磨清单①:
# 敌方替身/大招 HUD 不可见, 只能靠特效帧起表推算。
# self_dash: 自己按下 dash 后起表（真机 dash 无 HUD 区域, 同①走推算路线）
EVENT_CD_S: dict[str, float] = {
    "enemy_sub": 40.0, "self_sub": 40.0, "enemy_ult": 45.0,
    "self_dash": 10.0,
}

# 敌人失踪后位置置信度衰减时间常数（秒）—— P2#4:
# 敌人不可见时保留最后已知位置, 置信度 exp(-age/τ) 随时间衰减
ENEMY_POS_TAU_S = 1.0

# 威胁判定距离（px）: 敌方大招就绪且 dist<THREAT_RANGE_PX → relative.threat=1
THREAT_RANGE_PX = 400.0

# ───────────── ⑧ 检测器扩类: 威胁模型 / 墙体（施工⑧） ─────────────
# 命中判定半径(px): 弹道与自身中心的最小 miss 距离, 小于此值才算「会打中」
THREAT_HIT_RADIUS_PX = 45.0
# 弹道最低速(px/s): 低于此值视为静止特效而非飞行道具, 不参与 TTC
THREAT_MIN_SPEED_PX_S = 100.0
# hitbox_warn(技能预警圈)的 TTC【占位值】: 预警框不带启动帧信息,
# 用敌方典型出招前摇估计, 真机标定后可按敌方当前动作帧表推算
HITBOX_WARN_TTC_S = 0.25
# TTC 归一化上限(秒): threat_ttc_s / 此值 → [0,1] 进状态; 0.0=无威胁
THREAT_TTC_NORM_S = 1.0

# ── 墙体/场地边界 ──
# YOLO 未检出 wall 类时的兜底场地: 画面四边内缩此像素当作边界
WALL_FALLBACK_MARGIN_PX = 30.0
# 墙距归一化(px): wall_dist / 此值 → [0,1], 1.0=远离墙
WALL_DIST_NORM_PX = 500.0

# ───────────────────────── 触点语义分配 ─────────────────────────
# 摇杆与按钮必须分路: 抢同一路触点会互相顶掉（边走边放技能时按钮把杆松开）。
CONTACT_JOYSTICK = 0          # 摇杆持久流（长生命周期, 转向不释放）
CONTACT_BUTTON = 1            # 按钮点按/长按（一次性）

# ───────────────────────── 摇杆持久流 ─────────────────────────
# 转向判定: 持久流靠在线更新方向, 不存在"重建" → 无需角度死区。
# 但方向变化极小时不必重算目标点, 用这个阈值省掉无效 move（单位: 归一化位移）。
STEER_EPS = 1e-6

# ───────────────────────── 频率规格 ─────────────────────────
DECISION_HZ = 20              # 决策 20Hz
FAST_HZ = 60                  # 感知快变量 60Hz
SLOW_HZ = 12                  # 感知慢变量 12Hz（5帧一读）
JOYSTICK_HZ = 20              # 摇杆 20Hz 连续流

DECISION_PERIOD_MS = 1000.0 / DECISION_HZ     # 50ms 节拍

# 感知慢变量降频比: FAST_HZ / SLOW_HZ
SLOW_EVERY = FAST_HZ // SLOW_HZ   # = 5

# 决策看门狗: 前向超时 → 沿用上帧动作
DECISION_TIMEOUT_MS = 20

# ───────────────────────── 延迟预算表（v1.0 文档第二节） ─────────────────────────
# (环节, 预算ms, 保命手段)
LATENCY_BUDGET: tuple[tuple[str, float, str], ...] = (
    ("模拟器渲染",           16.0, "物理税，不可压缩"),
    ("画面捕获(scrcpy流)",    10.0, "弃ADB截图(仅8FPS)"),
    ("感知推理(YOLOv8n+CV)",  15.0, "320x320输入; 慢变量5帧一读"),
    ("状态组装",              1.0, "-"),
    ("决策前向(ONNX FP16)",   15.0, "超时20ms沿用上帧动作(看门狗)"),
    ("按键注入(minitouch)",    1.0, "弃adb input tap(30-100ms)"),
    ("模拟器响应",            16.0, "物理税"),
)

# 文档口径 ~65ms 按决策前向典型值计; 表中决策取 15ms 上限(看门狗口径) → 最坏 74ms
BUDGET_TOTAL_MS = 65.0        # 合计 ~65ms（文档口径, 决策取典型值）
BUDGET_WORST_CASE_MS = 74.0   # 决策取 15ms 上限时的最坏合计
BUDGET_MARGIN_MS = 25.0       # 余量 ~25ms

# M1-M3 感知闭环本地可测子集（沙箱/开发机验证用）
MEASURABLE_STAGE_BUDGET = {
    "capture": 10.0,          # 画面捕获
    "fast_detect": 15.0,      # 感知推理（快变量，含CV与检测）
    "slow_cv": 15.0,          # 感知推理（慢变量读数）
    "state": 1.0,             # 状态组装
}

# ───────────────────────── 数据规格（宪法第六节） ─────────────────────────
BC_DATA_PER_MATCH_MB = 2.8
PPO_TRAJECTORY_BUFFER_MB = 50
DISK_TOTAL_GB = 5
MEM_TOTAL_MB = 500

# ───────────────────────── 合成测试画面规格 ─────────────────────────
FRAME_W = 1280
FRAME_H = 720

assert abs(sum(b for _, b, _ in LATENCY_BUDGET) - BUDGET_WORST_CASE_MS) < 0.01, \
    "延迟预算表(决策取上限口径)合计必须等于 74ms"
assert SLOW_EVERY == 5

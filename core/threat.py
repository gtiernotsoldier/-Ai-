# -*- coding: utf-8 -*-
"""M4.7 — 威胁模型 (threat) —— 打磨清单施工⑧ 的核心

两个子模型:
  ThreatTracker  飞行道具逐帧关联 + 最近逼近法算 TTC(含脱靶判定)
                 hitbox_warn(技能预警圈) 用占位 TTC(框不带启动帧信息)
  WallModel      场地墙(YOLO wall 类) → 人/墙距离 + 最近墙点(躲避选向/逼墙走位)

TTC 语义(最近逼近法):
  弹道 p(t) = p0 + v·t, 求与自身 s 的最小距离时刻:
      t* = clamp( -((p0-s)·v)/|v|² , 0, +∞ )
  最小距离 d* = |p0 + v·t* - s|
  d* ≤ HIT_RADIUS → 会打中, threat_ttc = t*; 否则弹道打不中 → 不算威胁。
  这与"逐帧看距离"不同: 侧向飞过的弹即使某帧距离很近也不误报。

威胁来向 threat_dir: 弹道速度方向(单位向量) / 预警圈指向自身的方向。
躲避方向由策略层取其垂直向量(见 core/strategy.py dodge)。

墙模型: walls 为空 → 兜底场地 = 画面四边内缩 WALL_FALLBACK_MARGIN_PX
（真机 YOLO wall 类标定前, 边界即场地边缘, 与"压场"的直觉一致）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.constants import (
    FRAME_H, FRAME_W,
    HITBOX_WARN_TTC_S, THREAT_HIT_RADIUS_PX, THREAT_MIN_SPEED_PX_S,
    WALL_FALLBACK_MARGIN_PX,
)
from core.detectors import Detection


# ═════════════════════════ 威胁汇总 ═════════════════════════
@dataclass(frozen=True)
class ThreatSummary:
    """本帧最紧迫威胁的快照(供 PerceptionResult / 状态向量)。"""
    active: bool                    # False = 场上无会命中自己的威胁
    kind: str = ""                  # "projectile" | "hitbox_warn"
    ttc_s: float = 0.0              # 预计碰撞剩余时间
    dir_x: float = 0.0              # 威胁来向(弹道飞行方向/预警→自身), 单位向量
    dir_y: float = 0.0


_NO_THREAT = ThreatSummary(active=False)


# ═════════════════════════ 弹道追踪 ═════════════════════════
class _Track:
    """单个道具的跨帧轨迹: 两帧观测即可估计速度。"""

    __slots__ = ("x", "y", "vx", "vy", "last_ts")

    def __init__(self, x: float, y: float, ts: float):
        self.x, self.y = x, y
        self.vx = self.vy = 0.0
        self.last_ts = ts


class ThreatTracker:
    """projectile 逐帧最近邻关联 → 速度 → TTC; hitbox_warn 即时威胁。

    纯内存计算(宪法: 闭环内零网络/零磁盘)。track 失联超过 ttl 即遗忘
    （道具消失/检测漏帧不会留下僵尸轨迹）。
    """

    TRACK_TTL_S = 0.30          # 轨迹失联遗忘时间(≈18 帧)
    MATCH_RADIUS_PX = 80.0      # 关联半径(60fps 下道具单帧位移 ~10px, 足够宽)

    def __init__(self):
        self._tracks: list[_Track] = []
        self._last_ts: float | None = None

    # ── 主更新: 每帧调用 ──
    def update(self, dets: list[Detection], self_pos: tuple[float, float],
               now: float) -> ThreatSummary:
        dt = (now - self._last_ts) if self._last_ts is not None else 0.0
        self._last_ts = now

        proj = [d for d in dets if d.cls == "projectile"]
        warns = [d for d in dets if d.cls == "hitbox_warn"]
        self._associate(proj, dt, now)

        best = _NO_THREAT
        # 飞行道具: 最近逼近 TTC + 脱靶判定
        for t in self._tracks:
            speed = math.hypot(t.vx, t.vy)
            if speed < THREAT_MIN_SPEED_PX_S:
                continue                      # 静止特效, 不是弹
            sx, sy = self_pos
            if speed < 1e-6:
                continue
            # t* = -((p-s)·v)/|v|², 截到 ≥0
            denom = speed * speed
            t_star = -(((t.x - sx) * t.vx + (t.y - sy) * t.vy) / denom)
            t_star = max(0.0, t_star)
            mx = t.x + t.vx * t_star - sx
            my = t.y + t.vy * t_star - sy
            miss = math.hypot(mx, my)
            if miss > THREAT_HIT_RADIUS_PX:
                continue                      # 打不中 → 不是威胁(不躲空枪)
            if best is _NO_THREAT or t_star < best.ttc_s:
                best = ThreatSummary(
                    active=True, kind="projectile", ttc_s=t_star,
                    dir_x=t.vx / speed, dir_y=t.vy / speed)
        # 预警圈: 判定未生效, 圈心接近自己才算; TTC 占位(框无启动帧信息)
        for w in warns:
            cx, cy, cw, ch = w.x, w.y, w.w, w.h
            wx, wy = cx + cw / 2.0, cy + ch / 2.0
            radius = max(cw, ch) / 2.0
            sx, sy = self_pos
            dx, dy = sx - wx, sy - wy
            dist = math.hypot(dx, dy)
            if dist > radius + THREAT_HIT_RADIUS_PX:
                continue                      # 圈没罩到自己
            if best is _NO_THREAT or HITBOX_WARN_TTC_S < best.ttc_s:
                nx, ny = (dx / dist, dy / dist) if dist > 1e-3 else (0.0, 0.0)
                best = ThreatSummary(active=True, kind="hitbox_warn",
                                     ttc_s=HITBOX_WARN_TTC_S,
                                     dir_x=nx, dir_y=ny)
        return best

    # ── 最近邻关联 ──
    def _associate(self, proj: list[Detection], dt: float, now: float) -> None:
        # 遗忘失联轨迹（道具消失/检测漏帧不留僵尸轨迹）
        self._tracks = [t for t in self._tracks
                        if now - t.last_ts <= self.TRACK_TTL_S]
        used: set[int] = set()
        for d in proj:
            px, py = d.x + d.w / 2.0, d.y + d.h / 2.0
            bi, bd = -1, self.MATCH_RADIUS_PX
            for i, t in enumerate(self._tracks):
                if i in used:
                    continue
                dist = math.hypot(px - t.x, py - t.y)
                if dist < bd:
                    bi, bd = i, dist
            if bi >= 0:
                t = self._tracks[bi]
                if dt > 1e-4:                 # 两帧观测才可估速度
                    t.vx = (px - t.x) / dt
                    t.vy = (py - t.y) / dt
                t.x, t.y, t.last_ts = px, py, now
                used.add(bi)
            else:
                self._tracks.append(_Track(px, py, now))


# ═════════════════════════ 墙体模型 ═════════════════════════
class WallModel:
    """wall 检测框 → 距墙距离 + 最近墙点; 无框时用画面边界兜底。"""

    def __init__(self):
        self._walls: tuple[tuple[float, float, float, float], ...] = ()
        self._from_dets = False

    def update(self, dets: list[Detection]) -> None:
        ws = tuple((d.x, d.y, d.w, d.h) for d in dets if d.cls == "wall")
        if ws:
            self._walls, self._from_dets = ws, True
        elif not self._from_dets:
            self._walls = ()                  # 每帧重估: 本帧没检出就退回边界兜底
        else:
            self._walls = ()                  # 曾经有墙但本帧没检出(漏检) → 兜底

    def _arena_bounds(self) -> tuple[float, float, float, float]:
        m = WALL_FALLBACK_MARGIN_PX
        return (m, m, FRAME_W - m, FRAME_H - m)

    def nearest_wall_point(self, pos: tuple[float, float],
                           ) -> tuple[tuple[float, float], float]:
        """返回 (最近墙点, 距离px)。墙=检测框(实心矩形); 无框时才用场地边界兜底。

        兜底不与检出的墙框竞争: 墙框在场内时应以墙框为准
        （帧内 YOLO 漏检 wall 类 → 该帧退回边界, 由调用侧的滤波平滑）。
        """
        best_pt: tuple[float, float] | None = None
        best_d = float("inf")
        for (x, y, w, h) in self._walls:
            pt = (min(max(pos[0], x), x + w), min(max(pos[1], y), y + h))
            d = math.hypot(pos[0] - pt[0], pos[1] - pt[1])
            if d < best_d:
                best_pt, best_d = pt, d
        if not self._walls:
            # 兜底边界: 人在场内 → 距离 = 到最近边的距离(不是到包围盒的 0!)
            bx0, by0, bx1, by1 = self._arena_bounds()
            dx_left, dx_right = pos[0] - bx0, bx1 - pos[0]
            dy_bottom, dy_top = pos[1] - by0, by1 - pos[1]
            if dx_left >= 0 and dx_right >= 0 and dy_bottom >= 0 and dy_top >= 0:
                dxb = min(dx_left, dx_right)
                dyb = min(dy_bottom, dy_top)
                if dxb <= dyb:
                    best_pt = (bx0 if dx_left <= dx_right else bx1, pos[1])
                    best_d = dxb
                else:
                    best_pt = (pos[0], by0 if dy_bottom <= dy_top else by1)
                    best_d = dyb
            else:                         # 场外(异常): 钳回边界
                best_pt = (min(max(pos[0], bx0), bx1),
                           min(max(pos[1], by0), by1))
                best_d = math.hypot(pos[0] - best_pt[0], pos[1] - best_pt[1])
        return (best_pt or pos), best_d

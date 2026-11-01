# -*- coding: utf-8 -*-
"""episode 数据契约测试。

重点守四条不变量（对应模块文档里的四条设计规则）:
  ① 时间基准唯一 —— 掉帧能被检测, 折扣按 Δt 算
  ② valid 显式 —— 无效帧不参与采样
  ③ 存全帧 —— 保持帧降权而非丢弃
  ④ 元数据随行 —— 版本/标定指纹随数据走
"""
from __future__ import annotations

import numpy as np
import pytest

from core.constants import STATE_DIM
from core.episode import (EPISODE_VERSION, Episode, EpisodeMeta, FRAME_GAP_S,
                          load_dir)
from core.policy import DURATION_BUCKETS_MS
from core.state import LAYOUT_VERSION


def _meta(eid: str = "ep_test") -> EpisodeMeta:
    return EpisodeMeta(episode_id=eid, started_at=0.0, fps_nominal=20.0)


def _state(v: float = 0.5) -> np.ndarray:
    return np.full(STATE_DIM, v, dtype=np.float32)


def _make(ts=(0.00, 0.05, 0.10, 0.15), actions=None, valid=None, rewards=None):
    ep = Episode(_meta())
    actions = actions or [(1, 1, 0, 0)] * len(ts)
    valid = valid or [True] * len(ts)
    rewards = rewards or [0.0] * len(ts)
    for i, t in enumerate(ts):
        ep.append(t, i, _state(0.1 * i), valid[i], actions[i], hold_idx=3,
                  reward=rewards[i])
    ep.finish("win")
    return ep


class TestMeta:
    def test_from_env_carries_version_and_hash(self):
        m = EpisodeMeta.from_env("ep_1")
        assert m.layout_version == LAYOUT_VERSION
        assert m.keymap_hash != ""          # 读不到也应是 "unknown" 而非空串
        assert m.git_sha != ""

    def test_json_roundtrip_preserves_resolution_tuple(self):
        m = EpisodeMeta(episode_id="x", started_at=1.0, resolution=(1920, 1080))
        m2 = EpisodeMeta.from_json(m.to_json())
        assert m2.resolution == (1920, 1080)
        assert isinstance(m2.resolution, tuple)


class TestAppend:
    def test_ts_must_increase(self):
        ep = Episode(_meta())
        ep.append(0.10, 0, _state(), True, (1, 1, 0, 0))
        with pytest.raises(ValueError, match="严格递增"):
            ep.append(0.10, 1, _state(), True, (1, 1, 0, 0))

    def test_hold_idx_bounds(self):
        ep = Episode(_meta())
        with pytest.raises(ValueError, match="hold_idx 越界"):
            ep.append(0.0, 0, _state(), True, (1, 1, 0, 0),
                      hold_idx=len(DURATION_BUCKETS_MS))

    def test_action_must_be_4(self):
        ep = Episode(_meta())
        with pytest.raises(ValueError, match="4 元组"):
            ep.append(0.0, 0, _state(), True, (1, 1, 0))

    def test_state_shape_enforced(self):
        ep = Episode(_meta())
        with pytest.raises(ValueError):
            ep.append(0.0, 0, np.zeros(10, dtype=np.float32), True, (1, 1, 0, 0))

    def test_len_and_lazy_attr(self):
        ep = _make()
        assert len(ep) == 4
        assert ep.state.shape == (4, STATE_DIM)      # 懒冻结后可直接读
        assert ep.ts.dtype == np.float32


class TestGaps:
    def test_detects_missing_frames(self):
        ep = _make(ts=(0.00, 0.05, 0.20, 0.25))
        gaps = ep.gaps()
        assert len(gaps) == 1
        idx, dt = gaps[0]
        assert idx == 1 and dt == pytest.approx(0.15)
        # 0.15 / 0.05 = 3 拍 → 缺 2 拍; 实有 4 帧, 应有 6 帧
        assert ep.frame_drop_rate() == pytest.approx(2 / 6, abs=1e-6)

    def test_no_gap_in_uniform_stream(self):
        ep = _make(ts=tuple(np.arange(10) * 0.05))
        assert ep.gaps() == []
        assert ep.frame_drop_rate() == 0.0

    def test_threshold_is_configurable(self):
        ep = _make(ts=(0.0, 0.06))
        assert ep.gaps() == []                       # 0.06 < FRAME_GAP_S
        assert len(ep.gaps(threshold_s=0.055)) == 1


class TestChangePoints:
    def test_first_frame_always_counted(self):
        ep = _make()
        np.testing.assert_array_equal(ep.change_points(), [0])

    def test_detects_action_switch(self):
        ep = _make(actions=[(1, 1, 0, 0), (1, 1, 0, 0),
                            (1, 2, 0, 0), (1, 2, 0, 0)])
        np.testing.assert_array_equal(ep.change_points(), [0, 2])

    def test_hold_change_also_counts(self):
        ep = Episode(_meta())
        ep.append(0.00, 0, _state(), True, (1, 1, 0, 0), hold_idx=0)
        ep.append(0.05, 1, _state(), True, (1, 1, 0, 0), hold_idx=3)
        np.testing.assert_array_equal(ep.change_points(), [0, 1])

    def test_direction_change_counts(self):
        ep = _make(ts=(0.0, 0.05), actions=[(0, 1, 0, 0), (0, 1, 0, 5)])
        np.testing.assert_array_equal(ep.change_points(), [0, 1])


class TestWindow:
    def test_window_selects_by_time_not_index(self):
        # 中间掉了 3 帧: 索引相邻的 0/1 实际隔了 0.2s
        ep = _make(ts=(0.00, 0.20, 0.25, 0.30))
        w = ep.window(0.30, span_s=0.15)
        np.testing.assert_array_equal(w, [1, 2, 3])   # 不含 0（距 0.30 已 0.3s）

    def test_window_excludes_invalid(self):
        ep = _make(valid=[True, False, True, True])
        w = ep.window(0.15, span_s=0.4)
        np.testing.assert_array_equal(w, [0, 2, 3])


class TestReturns:
    def test_matches_hand_computation(self):
        ep = _make(ts=(0.0, 0.05, 0.10), rewards=[1.0, 0.0, 2.0])
        g = ep.returns(gamma_per_s=0.98)
        d = 0.98 ** 0.05
        assert g[2] == pytest.approx(2.0)
        assert g[1] == pytest.approx(d * 2.0)
        assert g[0] == pytest.approx(1.0 + d * d * 2.0)

    def test_discount_uses_real_dt(self):
        """同样两帧, 间隔 0.05 vs 0.20 的折扣必须不同 —— 按帧数算会错。"""
        fast = _make(ts=(0.0, 0.05), rewards=[0.0, 1.0]).returns(0.98)[0]
        slow = _make(ts=(0.0, 0.20), rewards=[0.0, 1.0]).returns(0.98)[0]
        assert fast == pytest.approx(0.98 ** 0.05)
        assert slow == pytest.approx(0.98 ** 0.20)
        assert fast > slow

    def test_terminal_bonus_only_at_last(self):
        ep = _make(rewards=[0.0, 0.0, 0.0, 0.0])
        g = ep.returns(gamma_per_s=1.0, terminal_bonus=7.0)
        assert g[-1] == pytest.approx(7.0)
        assert g[0] == pytest.approx(7.0)      # γ=1 时全程同值

    def test_monotone_decay(self):
        ep = _make(ts=tuple(np.arange(20) * 0.05), rewards=[1.0] * 20)
        g = ep.returns(0.98)
        assert np.all(np.diff(g) <= 1e-6)      # 越早看到越多未来, G 越大


class TestSampleWeights:
    def test_hold_frames_downweighted_not_dropped(self):
        ep = _make(actions=[(1, 1, 0, 0)] * 4)
        w = ep.sample_weights(hold_weight=0.3)
        assert len(w) == 4                     # 不丢弃
        assert w[0] == pytest.approx(1.0)      # 变化点
        assert np.allclose(w[1:], 0.3)

    def test_invalid_frames_zero_weight(self):
        ep = _make(valid=[True, False, True, True])
        w = ep.sample_weights()
        assert w[1] == 0.0


class TestValidate:
    def test_clean_episode_passes(self):
        assert _make().validate() == []

    def test_state_out_of_range_caught(self):
        ep = Episode(_meta())
        bad = np.full(STATE_DIM, 1.5, dtype=np.float32)
        ep.append(0.0, 0, bad, True, (1, 1, 0, 0))
        assert any("越界" in e for e in ep.validate())

    def test_nan_caught(self):
        ep = Episode(_meta())
        bad = np.full(STATE_DIM, np.nan, dtype=np.float32)
        ep.append(0.0, 0, bad, True, (1, 1, 0, 0))
        assert any("NaN" in e for e in ep.validate())

    def test_direction_out_of_range_caught(self):
        ep = Episode(_meta())
        ep.append(0.0, 0, _state(), True, (1, 1, 0, 9))   # 合法 0..8
        assert any("direction" in e for e in ep.validate())

    def test_timebase_anomaly_caught(self):
        """无节拍器录出的一局: 100 帧只跨 0.1s, 比 20Hz 快 20 倍。"""
        ep = Episode(_meta())
        for i in range(100):
            ep.append(i * 0.001, i, _state(), True, (1, 1, 0, 0))
        ep.finish()
        assert any("时基异常" in e for e in ep.validate())

    def test_normal_timebase_passes(self):
        assert _make().validate() == []


class TestPersistence:
    def test_roundtrip_bit_identical(self, tmp_path):
        ep = _make(rewards=[0.1, -0.2, 0.3, 0.0])
        p = ep.save(tmp_path / "a.npz")
        back = Episode.load(p)
        np.testing.assert_array_equal(back.ts, ep.ts)
        np.testing.assert_array_equal(back.state, ep.state)
        np.testing.assert_array_equal(back.action, ep.action)
        np.testing.assert_array_equal(back.valid, ep.valid)
        np.testing.assert_array_equal(back.reward, ep.reward)
        assert back.done is True
        assert back.result == "win"
        assert back.meta.episode_id == ep.meta.episode_id

    def test_version_stamped(self, tmp_path):
        d = np.load(str(_make().save(tmp_path / "v.npz")))
        assert int(d["_episode_version"]) == EPISODE_VERSION

    def test_wrong_version_rejected(self, tmp_path):
        p = tmp_path / "bad.npz"
        np.savez_compressed(
            p, _episode_version=np.array(999, dtype=np.int16),
            meta=np.array(_meta().to_json()), ts=np.zeros(1, np.float32),
            frame_id=np.zeros(1, np.int32), state=np.zeros((1, STATE_DIM), np.float16),
            valid=np.ones(1, bool), action=np.zeros((1, 4), np.uint8),
            hold_idx=np.zeros(1, np.uint8), reward=np.zeros(1, np.float32),
            done=np.array(True))
        with pytest.raises(ValueError, match="格式版本"):
            Episode.load(p)

    def test_compressed_size_is_reasonable(self, tmp_path):
        """3200 帧一局应落在 1MB 以内 —— 这是「存全帧」可行的前提。"""
        ep = Episode(_meta())
        rng = np.random.default_rng(0)
        for i in range(3200):
            ep.append(i * 0.05, i, rng.random(STATE_DIM).astype(np.float32),
                      True, (1, 1, 0, 0), reward=float(rng.normal()))
        ep.finish("win")
        p = ep.save(tmp_path / "big.npz")
        assert p.stat().st_size < 1_100_000


class TestLoadDir:
    def test_skips_wrong_layout_version(self, tmp_path):
        ep = _make()
        ep.save(tmp_path / "ok.npz")
        # 伪造一个旧版本 layout 的 episode
        ep2 = _make()
        ep2.meta = EpisodeMeta(episode_id="old", started_at=0.0,
                               layout_version=LAYOUT_VERSION - 1)
        ep2.save(tmp_path / "old.npz")
        got = load_dir(tmp_path)
        assert [e.meta.episode_id for e in got] == ["ep_test"]

    def test_skips_corrupt_file(self, tmp_path):
        _make().save(tmp_path / "ok.npz")
        (tmp_path / "broken.npz").write_bytes(b"not a npz")
        got = load_dir(tmp_path)
        assert len(got) == 1


class TestSummary:
    def test_summary_fields(self):
        s = _make().summary()
        assert s["frames"] == 4
        assert s["change_points"] == 1
        assert s["layout_version"] == LAYOUT_VERSION
        assert 0.0 <= s["valid_ratio"] <= 1.0

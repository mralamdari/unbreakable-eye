"""
Tests for the HeatmapAccumulator (src/engine/heatmap.py).

Tests the core math — Gaussian kernel generation, position addition,
exponential decay, and normalization. These are pure numpy operations.
"""
import time
import numpy as np
import pytest
from src.engine.heatmap import HeatmapAccumulator


class TestHeatmapGaussianKernel:
    """_make_gaussian_kernel should produce a valid normalized kernel."""

    def test_kernel_shape(self):
        kernel = HeatmapAccumulator._make_gaussian_kernel(41, 40)
        assert kernel.shape == (41, 41)

    def test_kernel_is_normalized(self):
        kernel = HeatmapAccumulator._make_gaussian_kernel(41, 40)
        assert abs(kernel.sum() - 1.0) < 1e-6

    def test_kernel_center_is_peak(self):
        kernel = HeatmapAccumulator._make_gaussian_kernel(5, 2)
        center = kernel[2, 2]
        # Center should be the maximum value
        assert center == kernel.max()

    def test_kernel_is_symmetric(self):
        kernel = HeatmapAccumulator._make_gaussian_kernel(41, 40)
        assert np.allclose(kernel, kernel.T)

    def test_small_sigma_kernel(self):
        """Small sigma should still produce a valid normalized kernel."""
        kernel = HeatmapAccumulator._make_gaussian_kernel(3, 0.5)
        assert abs(kernel.sum() - 1.0) < 1e-6
        assert kernel.shape == (3, 3)


class TestHeatmapInitialization:
    """HeatmapAccumulator should initialize to a zero heatmap."""

    def test_zero_initial_state(self):
        hm = HeatmapAccumulator(width=100, height=80)
        assert hm.heatmap.shape == (80, 100)
        assert hm.heatmap.sum() == 0.0
        assert hm.total_weight == 0.0

    def test_initial_get_heatmap_returns_zero(self):
        hm = HeatmapAccumulator(width=100, height=80)
        result = hm.get_heatmap()
        assert result.shape == (80, 100)
        assert result.dtype == np.uint8
        assert result.max() == 0

    def test_dimensions_are_correct(self):
        hm = HeatmapAccumulator(width=640, height=480)
        assert hm.width == 640
        assert hm.height == 480


class TestHeatmapAddPosition:
    """Adding positions should accumulate heat properly."""

    def test_add_one_position(self):
        hm = HeatmapAccumulator(width=100, height=100)
        hm.add_position(50.0, 50.0)
        assert hm.total_weight == 1.0
        assert hm.heatmap.sum() > 0

    def test_add_multiple_positions_same_spot(self):
        hm = HeatmapAccumulator(width=100, height=100)
        base = time.time()
        for i in range(10):
            hm.add_position(50.0, 50.0, now=base + i * 0.001)
        # total_weight should be ~10.0 (tiny decay from the 1ms intervals)
        assert hm.total_weight == pytest.approx(10.0, rel=0.001)

    def test_add_position_clips_to_edge(self):
        """Position outside frame should clip to edge, not crash."""
        hm = HeatmapAccumulator(width=100, height=100)
        base = time.time()
        hm.add_position(-10.0, -10.0, now=base)
        hm.add_position(200.0, 200.0, now=base + 0.001)
        assert hm.total_weight == pytest.approx(2.0, rel=0.001)


class TestHeatmapDecay:
    """Exponential decay should reduce heatmap values over time."""

    def test_decay_reduces_heatmap_sum(self):
        hm = HeatmapAccumulator(width=100, height=100)
        base = time.time()
        hm.add_position(50.0, 50.0, now=base)
        before_sum = hm.heatmap.sum()
        hm._decay(now=base + 1.0)  # 1 second later
        after_sum = hm.heatmap.sum()
        assert after_sum < before_sum

    def test_decay_by_known_factor(self):
        """With HEATMAP_DECAY_RATE=0.5, after 1s the weight should halve."""
        hm = HeatmapAccumulator(width=100, height=100)
        import src.core.config as config
        original = config.settings.HEATMAP_DECAY_RATE
        config.settings.HEATMAP_DECAY_RATE = 0.5
        try:
            base = time.time()
            hm.add_position(50.0, 50.0, now=base)
            hm._decay(now=base + 1.0)
            assert hm.total_weight == pytest.approx(0.5, rel=0.01)
        finally:
            config.settings.HEATMAP_DECAY_RATE = original

    def test_negative_dt_does_not_decay(self):
        hm = HeatmapAccumulator(width=100, height=100)
        base = time.time()
        hm.add_position(50.0, 50.0, now=base)
        before = hm.heatmap.sum()
        hm._decay(now=base - 1.0)  # time went backwards
        assert hm.heatmap.sum() == before

    def test_zero_dt_does_not_decay(self):
        hm = HeatmapAccumulator(width=100, height=100)
        base = time.time()
        hm.add_position(50.0, 50.0, now=base)
        before = hm.heatmap.sum()
        hm._decay(now=base)  # same time
        assert hm.heatmap.sum() == before


class TestHeatmapGetHeatmap:
    """get_heatmap should return a normalized uint8 array."""

    def test_returns_uint8(self):
        hm = HeatmapAccumulator(width=100, height=80)
        hm.add_position(50.0, 50.0)
        result = hm.get_heatmap()
        assert result.dtype == np.uint8

    def test_normalized_between_0_and_255(self):
        hm = HeatmapAccumulator(width=100, height=80)
        base = time.time()
        for i in range(100):
            hm.add_position(50.0, 50.0, now=base + i * 0.001)
        result = hm.get_heatmap()
        assert result.min() >= 0
        assert result.max() <= 255

    def test_higher_activity_region_brighter(self):
        hm = HeatmapAccumulator(width=200, height=200)
        base = time.time()
        # Add 5 positions at (50,50) and 1 at (150,150)
        for i in range(5):
            hm.add_position(50.0, 50.0, now=base + i * 0.001)
        hm.add_position(150.0, 150.0, now=base + 0.1)
        result = hm.get_heatmap()
        # Center pixel of the (50,50) group should be >= the (150,150) pixel
        # (use >= because with small counts they could both be 0 after uint8 cast)
        raw_50 = hm.heatmap[50, 50]
        raw_150 = hm.heatmap[150, 150]
        assert raw_50 > raw_150, f"raw heatmap: {raw_50} vs {raw_150}"

    def test_heatmap_values_raw_increase_with_more_activity(self):
        """Raw (pre-normalization) heatmap values should increase with activity."""
        hm = HeatmapAccumulator(width=100, height=100)
        base = time.time()
        hm.add_position(50.0, 50.0, now=base)
        v1 = hm.heatmap[50, 50]
        hm.add_position(50.0, 50.0, now=base + 0.001)
        v2 = hm.heatmap[50, 50]
        assert v2 > v1


class TestHeatmapReset:
    """Reset should clear the heatmap to zero."""

    def test_reset_clears_values(self):
        hm = HeatmapAccumulator(width=100, height=80)
        hm.add_position(50.0, 50.0)
        hm.reset()
        assert hm.heatmap.sum() == 0.0
        assert hm.total_weight == 0.0

    def test_reset_then_add_works(self):
        hm = HeatmapAccumulator(width=100, height=80)
        hm.add_position(50.0, 50.0)
        hm.reset()
        hm.add_position(60.0, 60.0)
        assert hm.total_weight == 1.0
        assert hm.heatmap.sum() > 0

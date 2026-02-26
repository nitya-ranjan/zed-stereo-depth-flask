"""Tests for the depth_processing module."""

import numpy as np
import pytest

from app.depth_processing import (
    compute_depth_stats,
    depth_to_histogram,
    detect_obstacles,
    encode_frame_jpeg,
    region_depth,
)


@pytest.fixture
def simple_depth():
    """A 100x100 depth map with values from 1.0 to 10.0."""
    return np.linspace(1.0, 10.0, 100 * 100).reshape(100, 100).astype(np.float32)


@pytest.fixture
def depth_with_nans():
    """A depth map that includes NaN (invalid) pixels."""
    d = np.full((50, 50), 5.0, dtype=np.float32)
    d[10:20, 10:20] = np.nan
    return d


class TestComputeDepthStats:
    def test_basic_stats(self, simple_depth):
        stats = compute_depth_stats(simple_depth)
        assert stats["min"] == pytest.approx(1.0, abs=0.01)
        assert stats["max"] == pytest.approx(10.0, abs=0.01)
        assert stats["mean"] == pytest.approx(5.5, abs=0.1)
        assert stats["valid_ratio"] == 1.0

    def test_with_nans(self, depth_with_nans):
        stats = compute_depth_stats(depth_with_nans)
        assert stats["mean"] == pytest.approx(5.0, abs=0.01)
        assert stats["valid_ratio"] < 1.0

    def test_all_nan(self):
        d = np.full((10, 10), np.nan, dtype=np.float32)
        stats = compute_depth_stats(d)
        assert stats["min"] is None
        assert stats["valid_ratio"] == 0.0


class TestRegionDepth:
    def test_centre_region(self, simple_depth):
        val = region_depth(simple_depth, 50, 50, size=5)
        assert val is not None
        assert 1.0 <= val <= 10.0

    def test_edge_region(self, simple_depth):
        val = region_depth(simple_depth, 0, 0, size=5)
        assert val is not None

    def test_nan_region(self):
        d = np.full((20, 20), np.nan, dtype=np.float32)
        assert region_depth(d, 10, 10) is None


class TestDetectObstacles:
    def test_no_obstacles(self):
        d = np.full((100, 100), 5.0, dtype=np.float32)
        results = detect_obstacles(d, threshold=1.5, grid_rows=2, grid_cols=2)
        assert len(results) == 4
        assert all(not r["is_obstacle"] for r in results)

    def test_all_obstacles(self):
        d = np.full((100, 100), 0.5, dtype=np.float32)
        results = detect_obstacles(d, threshold=1.5, grid_rows=2, grid_cols=2)
        assert all(r["is_obstacle"] for r in results)

    def test_partial_obstacles(self):
        d = np.full((100, 100), 5.0, dtype=np.float32)
        d[:50, :50] = 0.5
        results = detect_obstacles(d, threshold=1.5, grid_rows=2, grid_cols=2)
        obstacle_count = sum(1 for r in results if r["is_obstacle"])
        assert 0 < obstacle_count < 4


class TestEncodeFrameJpeg:
    def test_encode(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = (128, 64, 32)
        data = encode_frame_jpeg(img, quality=50)
        assert data is not None
        assert len(data) > 0
        # JPEG magic bytes
        assert data[:2] == b"\xff\xd8"

    def test_encode_with_resize(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        data = encode_frame_jpeg(img, quality=50, width=100, height=50)
        assert data is not None


class TestDepthToHistogram:
    def test_basic_histogram(self, simple_depth):
        h = depth_to_histogram(simple_depth, bins=10)
        assert len(h["counts"]) == 10
        assert len(h["edges"]) == 11
        assert sum(h["counts"]) > 0

    def test_empty_histogram(self):
        d = np.full((10, 10), np.nan, dtype=np.float32)
        h = depth_to_histogram(d)
        assert h["counts"] == []
        assert h["edges"] == []

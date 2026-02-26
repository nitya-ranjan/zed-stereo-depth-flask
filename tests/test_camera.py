"""Tests for the camera module."""

import time

import numpy as np
import pytest

from app.camera import DepthFrame, ZEDCamera, _camera_lock
import app.camera as camera_module


@pytest.fixture(autouse=True)
def reset_camera_singleton():
    with _camera_lock:
        camera_module._camera_instance = None
    yield
    with _camera_lock:
        if camera_module._camera_instance is not None:
            camera_module._camera_instance.close()
            camera_module._camera_instance = None


class TestDepthFrame:
    def test_creation(self):
        color = np.zeros((10, 10, 3), dtype=np.uint8)
        depth = np.ones((10, 10), dtype=np.float32)
        vis = np.zeros((10, 10, 3), dtype=np.uint8)
        frame = DepthFrame(color, depth, vis)
        assert frame.color is color
        assert frame.depth is depth
        assert frame.timestamp > 0

    def test_custom_timestamp(self):
        frame = DepthFrame(None, None, None, timestamp=42.0)
        assert frame.timestamp == 42.0


class TestZEDCamera:
    def test_simulated_capture(self):
        """Camera should produce frames in simulated mode."""
        config = {"ZED_RESOLUTION": "VGA", "ZED_FPS": 30}
        cam = ZEDCamera(config)
        cam.open()
        # Wait for a frame
        deadline = time.time() + 3
        while cam.latest_frame is None and time.time() < deadline:
            time.sleep(0.05)
        assert cam.latest_frame is not None
        assert cam.latest_frame.color is not None
        assert cam.latest_frame.depth is not None
        assert cam.latest_frame.depth_colorized is not None
        cam.close()
        assert not cam.is_running

    def test_get_depth_at(self):
        config = {"ZED_RESOLUTION": "VGA", "ZED_FPS": 30}
        cam = ZEDCamera(config)
        # Inject a frame manually
        depth = np.full((100, 100), 2.5, dtype=np.float32)
        cam._latest_frame = DepthFrame(
            np.zeros((100, 100, 3), dtype=np.uint8),
            depth,
            np.zeros((100, 100, 3), dtype=np.uint8),
        )
        assert cam.get_depth_at(50, 50) == 2.5
        assert cam.get_depth_at(200, 200) is None  # out of bounds
        assert cam.get_depth_at(-1, 0) is None

    def test_get_depth_at_nan(self):
        config = {}
        cam = ZEDCamera(config)
        depth = np.full((10, 10), np.nan, dtype=np.float32)
        cam._latest_frame = DepthFrame(None, depth, None)
        assert cam.get_depth_at(5, 5) is None

    def test_get_depth_at_no_frame(self):
        config = {}
        cam = ZEDCamera(config)
        assert cam.get_depth_at(0, 0) is None

"""Tests for the Flask routes / API endpoints."""

import json
import time
import threading

import numpy as np
import pytest

from app import create_app
from app.camera import DepthFrame, ZEDCamera, _camera_lock, get_camera
from app.detector import Detection
import app.camera as camera_module


@pytest.fixture(autouse=True)
def reset_camera_singleton():
    """Reset the camera singleton between tests."""
    with _camera_lock:
        camera_module._camera_instance = None
    yield
    with _camera_lock:
        if camera_module._camera_instance is not None:
            camera_module._camera_instance.close()
            camera_module._camera_instance = None


@pytest.fixture
def client():
    app = create_app("testing")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def client_with_frame(client):
    """A test client whose camera already has a frame loaded."""
    with client.application.app_context():
        cam = get_camera(client.application.config)
        # Inject a fake frame without starting the capture thread
        fake_color = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_depth = np.full((100, 100), 3.0, dtype=np.float32)
        fake_depth_vis = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_annotated = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_detections = [
            Detection(0, "person", 0.95, (10, 10, 60, 80), 3.0),
            Detection(2, "car", 0.82, (50, 30, 90, 70), 5.2),
        ]
        cam._latest_frame = DepthFrame(
            fake_color, fake_depth, fake_depth_vis,
            detections=fake_detections, annotated=fake_annotated,
        )
        cam._running = True
    return client


class TestIndexPage:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"ZED Stereo Depth Viewer" in resp.data


class TestAPIStatus:
    def test_status_returns_json(self, client_with_frame):
        resp = client_with_frame.get("/api/status")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "running" in data
        assert "has_frame" in data


class TestAPIDepthStats:
    def test_depth_stats(self, client_with_frame):
        resp = client_with_frame.get("/api/depth/stats")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "mean" in data
        assert data["mean"] == pytest.approx(3.0, abs=0.01)


class TestAPIDepthAt:
    def test_valid_point(self, client_with_frame):
        resp = client_with_frame.get("/api/depth/at?x=50&y=50")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["depth_m"] == pytest.approx(3.0, abs=0.01)

    def test_missing_params(self, client_with_frame):
        resp = client_with_frame.get("/api/depth/at")
        assert resp.status_code == 400


class TestAPIDepthRegion:
    def test_region_depth(self, client_with_frame):
        resp = client_with_frame.get("/api/depth/region?x=50&y=50&size=5")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["depth_m"] == pytest.approx(3.0, abs=0.01)


class TestAPIObstacles:
    def test_obstacle_grid(self, client_with_frame):
        resp = client_with_frame.get("/api/depth/obstacles?threshold=5.0&rows=2&cols=2")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert len(data["grid"]) == 4
        # All cells should be obstacles since depth=3.0 < threshold=5.0
        assert all(c["is_obstacle"] for c in data["grid"])


class TestAPIHistogram:
    def test_histogram(self, client_with_frame):
        resp = client_with_frame.get("/api/depth/histogram?bins=10")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert len(data["counts"]) == 10


class TestAPISnapshots:
    def test_color_snapshot(self, client_with_frame):
        resp = client_with_frame.get("/api/snapshot/color")
        assert resp.status_code == 200
        assert resp.content_type == "image/jpeg"
        assert resp.data[:2] == b"\xff\xd8"

    def test_depth_snapshot(self, client_with_frame):
        resp = client_with_frame.get("/api/snapshot/depth")
        assert resp.status_code == 200
        assert resp.content_type == "image/jpeg"


class TestAPIDetections:
    def test_detections_list(self, client_with_frame):
        resp = client_with_frame.get("/api/detections")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["count"] == 2
        assert len(data["detections"]) == 2
        assert data["detections"][0]["class_name"] == "person"
        assert data["detections"][0]["distance_m"] == 3.0
        assert data["detections"][1]["class_name"] == "car"

    def test_detections_summary(self, client_with_frame):
        resp = client_with_frame.get("/api/detections/summary")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "person" in data
        assert data["person"]["count"] == 1
        assert data["person"]["nearest_m"] == 3.0
        assert "car" in data

    def test_detection_snapshot(self, client_with_frame):
        resp = client_with_frame.get("/api/snapshot/detections")
        assert resp.status_code == 200
        assert resp.content_type == "image/jpeg"

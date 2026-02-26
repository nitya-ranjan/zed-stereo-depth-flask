"""Tests for the YOLO object detection module."""

import numpy as np
import pytest

from app.detector import Detection, ObjectDetector, _box_color, get_detector
import app.detector as detector_module


@pytest.fixture(autouse=True)
def reset_detector_singleton():
    """Reset the detector singleton between tests."""
    detector_module._detector_instance = None
    yield
    detector_module._detector_instance = None


class TestDetection:
    def test_to_dict(self):
        det = Detection(0, "person", 0.92, (10, 20, 100, 200), 3.45)
        d = det.to_dict()
        assert d["class_name"] == "person"
        assert d["confidence"] == 0.92
        assert d["bbox"]["x1"] == 10
        assert d["distance_m"] == 3.45

    def test_to_dict_no_distance(self):
        det = Detection(1, "car", 0.75, (0, 0, 50, 50))
        d = det.to_dict()
        assert d["distance_m"] is None


class TestBoxColor:
    def test_returns_tuple(self):
        c = _box_color(0)
        assert isinstance(c, tuple)
        assert len(c) == 3

    def test_wraps_around(self):
        assert _box_color(0) == _box_color(10)


class TestObjectDetector:
    def test_estimate_distance_uniform(self):
        depth = np.full((100, 100), 5.0, dtype=np.float32)
        d = ObjectDetector._estimate_distance(depth, 20, 20, 80, 80)
        assert d == pytest.approx(5.0, abs=0.01)

    def test_estimate_distance_nan(self):
        depth = np.full((100, 100), np.nan, dtype=np.float32)
        d = ObjectDetector._estimate_distance(depth, 20, 20, 80, 80)
        assert d is None

    def test_estimate_distance_clamping(self):
        depth = np.full((50, 50), 2.0, dtype=np.float32)
        # bbox extends outside image — should not crash
        d = ObjectDetector._estimate_distance(depth, -10, -10, 100, 100)
        assert d == pytest.approx(2.0, abs=0.01)

    def test_estimate_distance_zero_box(self):
        depth = np.full((50, 50), 3.0, dtype=np.float32)
        # degenerate box
        d = ObjectDetector._estimate_distance(depth, 25, 25, 25, 25)
        assert d is None

    def test_draw_detections_returns_copy(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Detection(0, "person", 0.9, (10, 10, 50, 50), 2.5)
        result = ObjectDetector.draw_detections(img, [det])
        assert result is not img
        assert result.shape == img.shape

    def test_draw_detections_empty(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = ObjectDetector.draw_detections(img, [])
        assert np.array_equal(result, img)

    def test_draw_detections_no_distance(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Detection(2, "cat", 0.8, (5, 5, 40, 40), None)
        result = ObjectDetector.draw_detections(img, [det])
        assert result is not img

    def test_detect_runs_with_model(self):
        """Integration test: run YOLO on a small synthetic image."""
        detector = ObjectDetector(model_name="yolo11n.pt", confidence=0.25)
        img = np.random.randint(0, 255, (320, 480, 3), dtype=np.uint8)
        depth = np.full((320, 480), 5.0, dtype=np.float32)
        detections = detector.detect(img, depth)
        # We can't guarantee detections on random noise, but it should not crash
        assert isinstance(detections, list)

    def test_detect_without_depth(self):
        """YOLO should still work without depth data."""
        detector = ObjectDetector(model_name="yolo11n.pt", confidence=0.25)
        img = np.random.randint(0, 255, (320, 480, 3), dtype=np.uint8)
        detections = detector.detect(img, depth=None)
        assert isinstance(detections, list)
        for det in detections:
            assert det.distance_m is None


class TestGetDetector:
    def test_singleton(self):
        d1 = get_detector()
        d2 = get_detector()
        assert d1 is d2

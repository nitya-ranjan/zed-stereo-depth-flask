"""YOLO object detection with depth distance estimation.

Runs a YOLO model on colour frames, then cross-references each detected
bounding box with the depth map to estimate how far each object is.
"""

import threading
import time

import cv2
import numpy as np

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class Detection:
    """A single detected object with distance information."""

    __slots__ = ("class_id", "class_name", "confidence", "bbox", "distance_m")

    def __init__(self, class_id, class_name, confidence, bbox, distance_m=None):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2) in pixels
        self.distance_m = distance_m

    def to_dict(self):
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": {
                "x1": self.bbox[0],
                "y1": self.bbox[1],
                "x2": self.bbox[2],
                "y2": self.bbox[3],
            },
            "distance_m": self.distance_m,
        }


# ── Colour palette for drawing boxes ────────────────────────────
_PALETTE = [
    (0, 255, 128),
    (255, 128, 0),
    (128, 0, 255),
    (0, 200, 255),
    (255, 0, 128),
    (128, 255, 0),
    (255, 255, 0),
    (0, 128, 255),
    (255, 0, 255),
    (0, 255, 255),
]


def _box_color(class_id):
    return _PALETTE[class_id % len(_PALETTE)]


class ObjectDetector:
    """Wraps a YOLO model for inference on frames.

    Parameters
    ----------
    model_name : str
        YOLO model variant, e.g. "yolo11n.pt", "yolov8s.pt".
    confidence : float
        Minimum confidence threshold (0-1).
    device : str
        Torch device ("cpu", "cuda", "cuda:0", etc.).
    """

    def __init__(self, model_name="yolo11n.pt", confidence=0.35, device="cpu"):
        self._model_name = model_name
        self._confidence = confidence
        self._device = device
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            if not YOLO_AVAILABLE:
                raise RuntimeError(
                    "ultralytics is not installed. "
                    "Run: pip install ultralytics"
                )
            self._model = YOLO(self._model_name)

    def detect(self, color_bgr, depth=None):
        """Run detection on a BGR image.

        Parameters
        ----------
        color_bgr : np.ndarray
            BGR image (H, W, 3).
        depth : np.ndarray, optional
            Depth map (H, W) in metres, used for distance estimation.

        Returns
        -------
        list[Detection]
        """
        with self._lock:
            self._ensure_model()
            results = self._model.predict(
                color_bgr,
                conf=self._confidence,
                device=self._device,
                verbose=False,
            )

        detections = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, f"class_{cls_id}")

            distance = None
            if depth is not None:
                distance = self._estimate_distance(depth, x1, y1, x2, y2)

            detections.append(
                Detection(cls_id, cls_name, conf, (x1, y1, x2, y2), distance)
            )

        return detections

    @staticmethod
    def _estimate_distance(depth, x1, y1, x2, y2):
        """Estimate distance to an object using the central region of its bbox.

        Uses the median of the central 50% of the bounding box to be
        robust to noisy edges and mixed foreground/background pixels.
        """
        h, w = depth.shape[:2]
        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        # Use central 50% of the box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = (x2 - x1) * 0.25
        bh = (y2 - y1) * 0.25
        rx1 = int(max(0, cx - bw))
        rx2 = int(min(w, cx + bw))
        ry1 = int(max(0, cy - bh))
        ry2 = int(min(h, cy + bh))

        if rx1 >= rx2 or ry1 >= ry2:
            return None

        region = depth[ry1:ry2, rx1:rx2]
        valid = region[np.isfinite(region)]
        if valid.size == 0:
            return None
        return round(float(np.median(valid)), 2)

    @staticmethod
    def draw_detections(image, detections):
        """Draw bounding boxes and distance labels on a BGR image.

        Returns a new image (does not modify the input).
        """
        img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = _box_color(det.class_id)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Build label
            label = f"{det.class_name} {det.confidence:.0%}"
            if det.distance_m is not None:
                label += f" | {det.distance_m:.1f}m"

            # Label background
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img,
                (x1, y1 - th - baseline - 4),
                (x1 + tw + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                img,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return img


# ── Module-level singleton ──────────────────────────────────────

_detector_instance = None
_detector_lock = threading.Lock()


def get_detector(model_name="yolo11n.pt", confidence=0.35, device="cpu"):
    """Return the singleton ObjectDetector."""
    global _detector_instance
    with _detector_lock:
        if _detector_instance is None:
            _detector_instance = ObjectDetector(model_name, confidence, device)
        return _detector_instance

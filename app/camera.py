"""ZED camera interface with fallback to simulated stereo depth.

When the ZED SDK (pyzed) is available, this module uses the real camera.
Otherwise it falls back to a simulated depth source built with OpenCV
and NumPy so the application can run on any machine for development.
"""

import threading
import time

import cv2
import numpy as np

try:
    import pyzed.sl as sl

    ZED_SDK_AVAILABLE = True
except ImportError:
    ZED_SDK_AVAILABLE = False

_RESOLUTION_MAP = {
    "HD2K": (2208, 1242),
    "HD1080": (1920, 1080),
    "HD720": (1280, 720),
    "VGA": (672, 376),
}

_DEPTH_MODE_MAP: dict = {}
if ZED_SDK_AVAILABLE:
    _DEPTH_MODE_MAP = {
        "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
        "QUALITY": sl.DEPTH_MODE.QUALITY,
        "ULTRA": sl.DEPTH_MODE.ULTRA,
        "NEURAL": sl.DEPTH_MODE.NEURAL,
    }


class DepthFrame:
    """Container for a single captured frame pair."""

    __slots__ = (
        "color",
        "depth",
        "depth_colorized",
        "point_cloud",
        "timestamp",
        "detections",
        "annotated",
    )

    def __init__(
        self,
        color,
        depth,
        depth_colorized,
        point_cloud=None,
        timestamp=None,
        detections=None,
        annotated=None,
    ):
        self.color = color
        self.depth = depth
        self.depth_colorized = depth_colorized
        self.point_cloud = point_cloud
        self.timestamp = timestamp or time.time()
        self.detections = detections or []
        self.annotated = annotated  # colour image with detection boxes drawn


class ZEDCamera:
    """Manages the ZED camera lifecycle and frame capture."""

    def __init__(self, config):
        self._config = config
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._latest_frame = None
        self._camera = None
        self._detector = None

    # -- public API -----------------------------------------------------------

    @property
    def is_running(self):
        return self._running

    @property
    def latest_frame(self):
        with self._lock:
            return self._latest_frame

    def open(self):
        """Open the ZED camera (or start simulation)."""
        if self._running:
            return

        if ZED_SDK_AVAILABLE:
            self._open_zed()

        # Lazy-init the YOLO detector
        self._init_detector()

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _init_detector(self):
        """Initialise the YOLO detector if ultralytics is available."""
        try:
            from app.detector import get_detector

            model = self._config.get("YOLO_MODEL", "yolo11n.pt")
            conf = self._config.get("YOLO_CONFIDENCE", 0.35)
            device = self._config.get("YOLO_DEVICE", "cpu")
            self._detector = get_detector(model, conf, device)
        except Exception:
            self._detector = None

    def _run_detection(self, color, depth):
        """Run YOLO on a frame and return (detections, annotated_image)."""
        if self._detector is None:
            return [], None
        try:
            from app.detector import ObjectDetector

            detections = self._detector.detect(color, depth)
            annotated = ObjectDetector.draw_detections(color, detections)
            return detections, annotated
        except Exception:
            return [], None

    def close(self):
        """Release the camera resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        if self._camera is not None and ZED_SDK_AVAILABLE:
            self._camera.close()
            self._camera = None

    def get_depth_at(self, x, y):
        """Return the depth value in metres at pixel (x, y)."""
        frame = self.latest_frame
        if frame is None or frame.depth is None:
            return None
        h, w = frame.depth.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            val = float(frame.depth[y, x])
            if np.isfinite(val):
                return round(val, 3)
        return None

    # -- ZED SDK helpers ------------------------------------------------------

    def _open_zed(self):
        self._camera = sl.Camera()
        init_params = sl.InitParameters()

        res_name = self._config.get("ZED_RESOLUTION", "HD720")
        res_attr = f"RESOLUTION_{res_name}" if hasattr(sl, f"RESOLUTION_{res_name}") else None
        if res_attr:
            init_params.camera_resolution = getattr(sl, res_attr)

        init_params.camera_fps = self._config.get("ZED_FPS", 30)

        depth_mode_name = self._config.get("ZED_DEPTH_MODE", "PERFORMANCE")
        if depth_mode_name in _DEPTH_MODE_MAP:
            init_params.depth_mode = _DEPTH_MODE_MAP[depth_mode_name]

        init_params.depth_minimum_distance = self._config.get("DEPTH_MIN_DISTANCE", 0.3)
        init_params.depth_maximum_distance = self._config.get("DEPTH_MAX_DISTANCE", 20.0)
        init_params.coordinate_units = sl.UNIT.METER

        status = self._camera.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")

    # -- capture loop ---------------------------------------------------------

    def _capture_loop(self):
        if ZED_SDK_AVAILABLE and self._camera is not None:
            self._capture_loop_zed()
        else:
            self._capture_loop_simulated()

    def _capture_loop_zed(self):
        runtime = sl.RuntimeParameters()
        image_mat = sl.Mat()
        depth_mat = sl.Mat()
        depth_vis_mat = sl.Mat()
        point_cloud_mat = sl.Mat()

        while self._running:
            if self._camera.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                self._camera.retrieve_image(image_mat, sl.VIEW.LEFT)
                self._camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                self._camera.retrieve_image(depth_vis_mat, sl.VIEW.DEPTH)
                self._camera.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)

                color = image_mat.get_data()[:, :, :3].copy()
                depth = depth_mat.get_data().copy()
                depth_vis = depth_vis_mat.get_data()[:, :, :3].copy()
                pc = point_cloud_mat.get_data().copy()

                detections, annotated = self._run_detection(color, depth)
                frame = DepthFrame(
                    color, depth, depth_vis, pc,
                    detections=detections, annotated=annotated,
                )
                with self._lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.001)

    def _capture_loop_simulated(self):
        """Generate synthetic depth frames for development without a ZED."""
        res_name = self._config.get("ZED_RESOLUTION", "HD720")
        w, h = _RESOLUTION_MAP.get(res_name, (1280, 720))
        fps = self._config.get("ZED_FPS", 30)
        interval = 1.0 / fps
        t = 0.0

        while self._running:
            t += interval
            color = self._make_simulated_color(w, h, t)
            depth = self._make_simulated_depth(w, h, t)
            depth_colorized = self._colorize_depth(depth)

            detections, annotated = self._run_detection(color, depth)
            frame = DepthFrame(
                color, depth, depth_colorized,
                detections=detections, annotated=annotated,
            )
            with self._lock:
                self._latest_frame = frame

            time.sleep(interval)

    # -- simulation helpers ---------------------------------------------------

    @staticmethod
    def _make_simulated_color(w, h, t):
        """Create a synthetic colour image with moving shapes."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (30, 30, 40)

        # Draw a grid
        for gx in range(0, w, 80):
            cv2.line(img, (gx, 0), (gx, h), (50, 50, 60), 1)
        for gy in range(0, h, 80):
            cv2.line(img, (0, gy), (w, gy), (50, 50, 60), 1)

        # Moving circles representing objects at different depths
        cx1 = int(w * 0.3 + 100 * np.sin(t * 0.5))
        cy1 = int(h * 0.4 + 60 * np.cos(t * 0.3))
        cv2.circle(img, (cx1, cy1), 60, (0, 180, 255), -1)

        cx2 = int(w * 0.7 + 80 * np.cos(t * 0.7))
        cy2 = int(h * 0.6 + 40 * np.sin(t * 0.4))
        cv2.rectangle(img, (cx2 - 50, cy2 - 50), (cx2 + 50, cy2 + 50), (255, 100, 0), -1)

        cx3 = int(w * 0.5 + 120 * np.sin(t * 0.2))
        cy3 = int(h * 0.3 + 30 * np.cos(t * 0.6))
        pts = np.array(
            [
                [cx3, cy3 - 40],
                [cx3 - 35, cy3 + 25],
                [cx3 + 35, cy3 + 25],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(img, [pts], (100, 255, 100))

        cv2.putText(
            img,
            "SIMULATED - No ZED Camera",
            (w // 2 - 180, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return img

    @staticmethod
    def _make_simulated_depth(w, h, t):
        """Create a synthetic depth map in metres."""
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        # Base depth: gradient from near to far
        depth = 1.0 + 9.0 * (y_coords / h)

        # Add some oscillating bumps
        cx1 = w * 0.3 + 100 * np.sin(t * 0.5)
        cy1 = h * 0.4 + 60 * np.cos(t * 0.3)
        d1 = np.sqrt((x_coords - cx1) ** 2 + (y_coords - cy1) ** 2)
        depth -= 4.0 * np.exp(-(d1**2) / (2 * 60**2))

        cx2 = w * 0.7 + 80 * np.cos(t * 0.7)
        cy2 = h * 0.6 + 40 * np.sin(t * 0.4)
        d2 = np.sqrt((x_coords - cx2) ** 2 + (y_coords - cy2) ** 2)
        depth -= 2.5 * np.exp(-(d2**2) / (2 * 50**2))

        return np.clip(depth, 0.3, 20.0).astype(np.float32)

    @staticmethod
    def _colorize_depth(depth, min_d=0.3, max_d=20.0):
        """Apply a colour map to a depth array for visualization."""
        normalised = np.clip((depth - min_d) / (max_d - min_d), 0, 1)
        grey = (255 * (1.0 - normalised)).astype(np.uint8)
        return cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO)


# -- module-level singleton ---------------------------------------------------

_camera_instance = None
_camera_lock = threading.Lock()


def get_camera(config) -> ZEDCamera:
    """Return the singleton ZEDCamera, creating it if needed."""
    global _camera_instance
    with _camera_lock:
        if _camera_instance is None:
            _camera_instance = ZEDCamera(config)
        return _camera_instance

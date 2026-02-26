"""Flask route definitions for the stereo depth application."""

import time

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
)

from app.camera import get_camera
from app.depth_processing import (
    compute_depth_stats,
    depth_to_histogram,
    detect_obstacles,
    encode_frame_jpeg,
    region_depth,
)

main_bp = Blueprint("main", __name__)
api_bp = Blueprint("api", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _camera():
    cam = get_camera(current_app.config)
    if not cam.is_running:
        cam.open()
    return cam


def _stream_quality():
    return current_app.config.get("STREAM_QUALITY", 70)


def _stream_size():
    return (
        current_app.config.get("STREAM_WIDTH", 640),
        current_app.config.get("STREAM_HEIGHT", 360),
    )


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@main_bp.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# MJPEG video streams
# ---------------------------------------------------------------------------

def _generate_mjpeg(frame_attr):
    """Yield MJPEG frames from the camera's latest capture."""
    cam = _camera()
    quality = _stream_quality()
    w, h = _stream_size()

    while True:
        frame = cam.latest_frame
        if frame is not None:
            img = getattr(frame, frame_attr)
            if img is not None:
                jpeg = encode_frame_jpeg(img, quality=quality, width=w, height=h)
                if jpeg:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
        time.sleep(0.033)  # ~30 fps cap


@main_bp.route("/video/color")
def video_color():
    return Response(
        _generate_mjpeg("color"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@main_bp.route("/video/depth")
def video_depth():
    return Response(
        _generate_mjpeg("depth_colorized"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@main_bp.route("/video/detections")
def video_detections():
    """MJPEG stream of colour feed with YOLO detection overlays."""
    return Response(
        _generate_mjpeg("annotated"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@api_bp.route("/status")
def api_status():
    cam = _camera()
    frame = cam.latest_frame
    return jsonify(
        {
            "running": cam.is_running,
            "has_frame": frame is not None,
            "timestamp": frame.timestamp if frame else None,
        }
    )


@api_bp.route("/depth/stats")
def api_depth_stats():
    cam = _camera()
    frame = cam.latest_frame
    if frame is None or frame.depth is None:
        return jsonify({"error": "no frame available"}), 503
    stats = compute_depth_stats(frame.depth)
    return jsonify(stats)


@api_bp.route("/depth/at")
def api_depth_at():
    x = request.args.get("x", type=int)
    y = request.args.get("y", type=int)
    if x is None or y is None:
        return jsonify({"error": "x and y query parameters required"}), 400
    cam = _camera()
    val = cam.get_depth_at(x, y)
    if val is None:
        return jsonify({"error": "invalid coordinates or no data"}), 404
    return jsonify({"x": x, "y": y, "depth_m": val})


@api_bp.route("/depth/region")
def api_depth_region():
    x = request.args.get("x", type=int)
    y = request.args.get("y", type=int)
    size = request.args.get("size", 10, type=int)
    if x is None or y is None:
        return jsonify({"error": "x and y query parameters required"}), 400
    cam = _camera()
    frame = cam.latest_frame
    if frame is None or frame.depth is None:
        return jsonify({"error": "no frame available"}), 503
    val = region_depth(frame.depth, x, y, size)
    return jsonify({"x": x, "y": y, "size": size, "depth_m": val})


@api_bp.route("/depth/histogram")
def api_depth_histogram():
    cam = _camera()
    frame = cam.latest_frame
    if frame is None or frame.depth is None:
        return jsonify({"error": "no frame available"}), 503
    bins = request.args.get("bins", 50, type=int)
    hist = depth_to_histogram(frame.depth, bins=bins)
    return jsonify(hist)


@api_bp.route("/depth/obstacles")
def api_depth_obstacles():
    cam = _camera()
    frame = cam.latest_frame
    if frame is None or frame.depth is None:
        return jsonify({"error": "no frame available"}), 503
    threshold = request.args.get("threshold", 1.5, type=float)
    rows = request.args.get("rows", 4, type=int)
    cols = request.args.get("cols", 6, type=int)
    results = detect_obstacles(frame.depth, threshold=threshold, grid_rows=rows, grid_cols=cols)
    return jsonify({"threshold": threshold, "grid": results})


@api_bp.route("/snapshot/color")
def api_snapshot_color():
    cam = _camera()
    frame = cam.latest_frame
    if frame is None or frame.color is None:
        return jsonify({"error": "no frame available"}), 503
    jpeg = encode_frame_jpeg(frame.color, quality=90)
    return Response(jpeg, mimetype="image/jpeg")


@api_bp.route("/snapshot/depth")
def api_snapshot_depth():
    cam = _camera()
    frame = cam.latest_frame
    if frame is None or frame.depth_colorized is None:
        return jsonify({"error": "no frame available"}), 503
    jpeg = encode_frame_jpeg(frame.depth_colorized, quality=90)
    return Response(jpeg, mimetype="image/jpeg")


# ---------------------------------------------------------------------------
# Detection API
# ---------------------------------------------------------------------------

@api_bp.route("/detections")
def api_detections():
    """Return the latest YOLO detections with distance estimates."""
    cam = _camera()
    frame = cam.latest_frame
    if frame is None:
        return jsonify({"error": "no frame available"}), 503
    return jsonify({
        "count": len(frame.detections),
        "detections": [d.to_dict() for d in frame.detections],
    })


@api_bp.route("/detections/summary")
def api_detections_summary():
    """Summarised detection counts grouped by class name."""
    cam = _camera()
    frame = cam.latest_frame
    if frame is None:
        return jsonify({"error": "no frame available"}), 503
    summary = {}
    for det in frame.detections:
        entry = summary.setdefault(det.class_name, {"count": 0, "nearest_m": None})
        entry["count"] += 1
        if det.distance_m is not None:
            if entry["nearest_m"] is None or det.distance_m < entry["nearest_m"]:
                entry["nearest_m"] = det.distance_m
    return jsonify(summary)


@api_bp.route("/snapshot/detections")
def api_snapshot_detections():
    """Single JPEG frame with detection bounding boxes drawn."""
    cam = _camera()
    frame = cam.latest_frame
    if frame is None or frame.annotated is None:
        return jsonify({"error": "no frame or detections available"}), 503
    jpeg = encode_frame_jpeg(frame.annotated, quality=90)
    return Response(jpeg, mimetype="image/jpeg")

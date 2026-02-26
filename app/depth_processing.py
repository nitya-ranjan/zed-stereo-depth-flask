"""Depth map processing utilities.

Provides helper functions for analysing depth data: statistics, region
extraction, obstacle detection, and depth-image encoding for streaming.
"""

import cv2
import numpy as np


def compute_depth_stats(depth):
    """Return basic statistics for a depth map (in metres).

    Parameters
    ----------
    depth : np.ndarray
        2-D float array of depth values.

    Returns
    -------
    dict with keys: min, max, mean, median, std, valid_ratio
    """
    valid = depth[np.isfinite(depth)]
    if valid.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "valid_ratio": 0.0,
        }
    return {
        "min": round(float(valid.min()), 3),
        "max": round(float(valid.max()), 3),
        "mean": round(float(valid.mean()), 3),
        "median": round(float(np.median(valid)), 3),
        "std": round(float(valid.std()), 3),
        "valid_ratio": round(valid.size / depth.size, 3),
    }


def region_depth(depth, x, y, size=10):
    """Average depth in a square region centred at (x, y)."""
    h, w = depth.shape[:2]
    x0 = max(0, x - size)
    x1 = min(w, x + size + 1)
    y0 = max(0, y - size)
    y1 = min(h, y + size + 1)
    region = depth[y0:y1, x0:x1]
    valid = region[np.isfinite(region)]
    if valid.size == 0:
        return None
    return round(float(valid.mean()), 3)


def detect_obstacles(depth, threshold=1.5, grid_rows=4, grid_cols=6):
    """Split the frame into a grid and flag cells closer than *threshold* m.

    Returns a list of dicts with row, col, mean_depth, is_obstacle.
    """
    h, w = depth.shape[:2]
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    results = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            cell = depth[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            valid = cell[np.isfinite(cell)]
            mean_d = float(valid.mean()) if valid.size > 0 else float("inf")
            results.append(
                {
                    "row": r,
                    "col": c,
                    "mean_depth": round(mean_d, 3),
                    "is_obstacle": mean_d < threshold,
                }
            )
    return results


def encode_frame_jpeg(frame_bgr, quality=70, width=None, height=None):
    """Resize (optionally) and JPEG-encode a BGR image.

    Returns raw JPEG bytes.
    """
    if width and height:
        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return buf.tobytes()


def depth_to_histogram(depth, bins=50, range_min=0.0, range_max=20.0):
    """Return a histogram of depth values.

    Returns
    -------
    dict with 'edges' (list[float]) and 'counts' (list[int]).
    """
    valid = depth[np.isfinite(depth)]
    if valid.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(valid, bins=bins, range=(range_min, range_max))
    return {
        "edges": [round(float(e), 3) for e in edges],
        "counts": [int(c) for c in counts],
    }

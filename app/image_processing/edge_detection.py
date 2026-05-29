"""
Graphite — edge_detection.py
Responsibility: Canny edge detection on a preprocessed grayscale image.
"""

import cv2
import numpy as np


# ── Defaults ──────────────────────────────────────────────────────────────────
# These thresholds are tunable. General guidance:
#   - Lower threshold_low  → picks up weaker edges (more noise)
#   - Higher threshold_high → only strong edges survive (may lose detail)
DEFAULT_LOW  = 50
DEFAULT_HIGH = 150


def detect_edges(
    blurred: np.ndarray,
    threshold_low: int  = DEFAULT_LOW,
    threshold_high: int = DEFAULT_HIGH,
) -> np.ndarray:
    """
    Apply Canny edge detection to a blurred grayscale image.

    Args:
        blurred:        2D NumPy array (uint8) — output of apply_gaussian_blur().
        threshold_low:  Lower hysteresis threshold. Default: 50.
        threshold_high: Upper hysteresis threshold. Default: 150.

    Returns:
        Binary edge map as a 2D NumPy array (uint8).
        Pixels are 255 (edge) or 0 (background).
    """
    if blurred.ndim != 2:
        raise ValueError(
            f"[edge_detection] Expected a 2D grayscale array, got shape {blurred.shape}."
        )

    if threshold_low >= threshold_high:
        raise ValueError(
            f"[edge_detection] threshold_low ({threshold_low}) must be "
            f"less than threshold_high ({threshold_high})."
        )

    edges = cv2.Canny(blurred, threshold_low, threshold_high)

    edge_px = int(np.count_nonzero(edges))
    total_px = edges.size
    coverage = (edge_px / total_px) * 100

    print(
        f"[edge_detection] Canny done — "
        f"low: {threshold_low}, high: {threshold_high} | "
        f"edge pixels: {edge_px} / {total_px} ({coverage:.1f}%)"
    )

    return edges
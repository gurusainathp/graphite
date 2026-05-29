"""
Graphite — contours.py
Responsibility: Extract contours from an edge map and convert to internal geometry models.
"""

import cv2
import numpy as np
from typing import Tuple

from geometry.models import Point, BoundingBox, Contour, ContourCollection


def extract_contours(edges: np.ndarray) -> ContourCollection:
    """
    Extract all contours from a Canny edge map.

    Uses RETR_TREE to capture full parent/child hierarchy (external + internal).
    Uses CHAIN_APPROX_NONE to keep every boundary point (important for curve accuracy).

    Args:
        edges: Binary edge map (uint8) from detect_edges().

    Returns:
        ContourCollection with all contours converted to internal models.
    """
    if edges.ndim != 2:
        raise ValueError(f"[contours] Expected 2D edge map, got shape {edges.shape}.")

    raw_contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_TREE,          # full parent-child hierarchy
        cv2.CHAIN_APPROX_NONE,  # keep all contour points
    )

    h, w = edges.shape
    collection = ContourCollection(image_width=w, image_height=h)

    if hierarchy is None or len(raw_contours) == 0:
        print("[contours] No contours found.")
        return collection

    # hierarchy shape: (1, N, 4) → flatten to (N, 4)
    # Each row: [next, prev, first_child, parent]
    hier = hierarchy[0]

    for idx, (raw, hier_entry) in enumerate(zip(raw_contours, hier)):
        parent = int(hier_entry[3])
        is_external = (parent == -1)

        area      = float(cv2.contourArea(raw))
        perimeter = float(cv2.arcLength(raw, closed=True))
        bx, by, bw, bh = cv2.boundingRect(raw)

        points = [Point(float(pt[0][0]), float(pt[0][1])) for pt in raw]

        contour = Contour(
            points       = points,
            area         = area,
            perimeter    = perimeter,
            bbox         = BoundingBox(float(bx), float(by), float(bw), float(bh)),
            is_external  = is_external,
            hierarchy_id = idx,
        )
        collection.contours.append(contour)

    print(f"[contours] Extracted {collection.count} contours "
          f"({len(collection.external)} external, {len(collection.internal)} internal)")

    return collection


def draw_contours(
    collection: ContourCollection,
    label: str = "all",
) -> np.ndarray:
    """
    Render contours onto a black canvas for debug visualization.

    Args:
        collection: ContourCollection to draw.
        label:      'all' draws everything; 'external' draws only outer contours.

    Returns:
        BGR image (uint8 NumPy array) with contours drawn.
    """
    canvas = np.zeros(
        (collection.image_height, collection.image_width, 3), dtype=np.uint8
    )

    source = collection.contours if label == "all" else collection.external

    for contour in source:
        # Convert back to OpenCV format for drawing
        raw = _contour_to_raw(contour)
        color = (0, 255, 100) if contour.is_external else (100, 100, 255)
        cv2.drawContours(canvas, [raw], -1, color, 1)

    drawn = sum(1 for _ in source)
    print(f"[contours] Drew {drawn} contours onto canvas (label='{label}')")
    return canvas


# ── Internal helpers ──────────────────────────────────────────────────────────

def _contour_to_raw(contour: Contour) -> np.ndarray:
    """Convert a Contour model back to an OpenCV-compatible array for drawing."""
    pts = np.array(
        [[[int(p.x), int(p.y)]] for p in contour.points], dtype=np.int32
    )
    return pts
"""
Graphite — simplification.py
Responsibility: Reduce noisy contour points using Douglas-Peucker (approxPolyDP).
Fewer points → cleaner geometry → simpler equations.
"""

import cv2
import numpy as np

from geometry.models import Point, Contour, ContourCollection, BoundingBox


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_EPSILON_RATIO = 0.01   # fraction of perimeter used as approx tolerance


def simplify_contour(
    contour: Contour,
    epsilon_ratio: float = DEFAULT_EPSILON_RATIO,
) -> Contour:
    """
    Simplify a single contour using Douglas-Peucker approximation.

    epsilon = epsilon_ratio * perimeter
    Larger epsilon → fewer points, coarser shape.
    Smaller epsilon → more points, finer shape.

    Args:
        contour:       Source Contour (not mutated).
        epsilon_ratio: Tolerance as a fraction of the contour perimeter.

    Returns:
        New Contour with reduced point set.
    """
    if contour.point_count < 3:
        return contour

    raw = np.array([[[int(p.x), int(p.y)]] for p in contour.points], dtype=np.int32)
    epsilon = epsilon_ratio * contour.perimeter

    approx = cv2.approxPolyDP(raw, epsilon=epsilon, closed=True)
    simplified_points = [Point(float(pt[0][0]), float(pt[0][1])) for pt in approx]

    new_raw   = np.array([[[int(p.x), int(p.y)]] for p in simplified_points], dtype=np.int32)
    new_area  = float(cv2.contourArea(new_raw))
    new_perim = float(cv2.arcLength(new_raw, closed=True))
    bx, by, bw, bh = cv2.boundingRect(new_raw)

    return Contour(
        points       = simplified_points,
        area         = new_area,
        perimeter    = new_perim,
        bbox         = BoundingBox(float(bx), float(by), float(bw), float(bh)),
        is_external  = contour.is_external,
        hierarchy_id = contour.hierarchy_id,
    )


def simplify_collection(
    collection: ContourCollection,
    epsilon_ratio: float = DEFAULT_EPSILON_RATIO,
) -> ContourCollection:
    """
    Simplify all contours in a collection.

    Args:
        collection:    Source ContourCollection (not mutated).
        epsilon_ratio: Passed through to simplify_contour().

    Returns:
        New ContourCollection with simplified contours.
    """
    simplified = [simplify_contour(c, epsilon_ratio) for c in collection.contours]

    before_pts = sum(c.point_count for c in collection.contours)
    after_pts  = sum(c.point_count for c in simplified)
    reduction  = (1 - after_pts / before_pts) * 100 if before_pts > 0 else 0

    print(f"[simplification] Contours : {collection.count}")
    print(f"[simplification] Points   : {before_pts} → {after_pts} ({reduction:.1f}% reduction)")

    return ContourCollection(
        contours     = simplified,
        image_width  = collection.image_width,
        image_height = collection.image_height,
    )


def draw_simplified(collection: ContourCollection) -> np.ndarray:
    """Render simplified contours with vertex dots so point reduction is visually obvious."""
    canvas = np.zeros(
        (collection.image_height, collection.image_width, 3), dtype=np.uint8
    )
    for contour in collection.contours:
        if contour.point_count < 2:
            continue
        pts = np.array(
            [[[int(p.x), int(p.y)]] for p in contour.points], dtype=np.int32
        )
        color = (0, 255, 120) if contour.is_external else (120, 120, 255)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=1)
        for p in contour.points:
            cv2.circle(canvas, (int(p.x), int(p.y)), 2, (255, 200, 0), -1)

    print(f"[simplification] Drew {collection.count} simplified contours")
    return canvas
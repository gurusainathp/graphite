"""
Graphite — normalization.py
Responsibility: Convert image pixel coordinates into centered, graph-friendly coordinates.

Image space:    origin top-left,  y increases downward,  range [0, W] x [0, H]
Desmos space:   origin centered,  y increases upward,    range [-1, 1] x [-1, 1] (aspect-corrected)

Transformation:
    x_norm = (x_px - cx) / scale
    y_norm = -(y_px - cy) / scale      ← y-flip: image y is inverted vs graph y

where:
    cx    = image_width  / 2
    cy    = image_height / 2
    scale = max(image_width, image_height) / 2
"""

import numpy as np
from typing import Tuple

from geometry.models import Point, Contour, ContourCollection


def normalize_collection(collection: ContourCollection) -> ContourCollection:
    """
    Return a new ContourCollection where every point is in normalized graph space.

    Coordinate system after normalization:
        - Origin (0, 0) is the image center.
        - X range: roughly [-1, 1] (wider axis).
        - Y range: same scale (aspect-preserved), positive = up.

    Args:
        collection: Source ContourCollection in pixel space (not mutated).

    Returns:
        New ContourCollection with normalized Point coordinates.
        BoundingBox values are also normalized.
        image_width / image_height retain the originals for reference.
    """
    w = collection.image_width
    h = collection.image_height

    if w == 0 or h == 0:
        raise ValueError("[normalization] ContourCollection has zero image dimensions.")

    cx    = w / 2.0
    cy    = h / 2.0
    scale = max(w, h) / 2.0

    normalized_contours = []

    for c in collection.contours:
        norm_points = [_normalize_point(p, cx, cy, scale) for p in c.points]

        # Normalize bounding box corners then reconstruct
        tl = _normalize_point(Point(c.bbox.x, c.bbox.y), cx, cy, scale)
        br = _normalize_point(
            Point(c.bbox.x + c.bbox.width, c.bbox.y + c.bbox.height), cx, cy, scale
        )
        from geometry.models import BoundingBox
        norm_bbox = BoundingBox(
            x      = min(tl.x, br.x),
            y      = min(tl.y, br.y),
            width  = abs(br.x - tl.x),
            height = abs(br.y - tl.y),
        )

        # Scale area and perimeter to normalized units
        norm_area      = c.area      / (scale ** 2)
        norm_perimeter = c.perimeter / scale

        normalized_contours.append(Contour(
            points       = norm_points,
            area         = norm_area,
            perimeter    = norm_perimeter,
            bbox         = norm_bbox,
            is_external  = c.is_external,
            hierarchy_id = c.hierarchy_id,
        ))

    result = ContourCollection(
        contours     = normalized_contours,
        image_width  = w,
        image_height = h,
    )

    xs = [p.x for c in result.contours for p in c.points]
    ys = [p.y for c in result.contours for p in c.points]
    if xs and ys:
        print(f"[normalization] Normalized {result.count} contours")
        print(f"[normalization] X range: [{min(xs):.3f}, {max(xs):.3f}]")
        print(f"[normalization] Y range: [{min(ys):.3f}, {max(ys):.3f}]")

    return result


def render_normalized_preview(
    norm_collection: ContourCollection,
    output_size: int = 800,
) -> np.ndarray:
    """
    Render normalized contours onto a fixed-size canvas for visual verification.

    Args:
        norm_collection: ContourCollection in normalized [-1, 1] coordinate space.
        output_size:     Canvas size in pixels (square). Default: 800.

    Returns:
        BGR uint8 NumPy image.
    """
    canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    pad    = 40  # pixel padding from canvas edges

    draw_range = output_size - 2 * pad
    half       = output_size / 2.0

    # Draw centered axes
    cv_gray = (60, 60, 60)
    import cv2
    cv2.line(canvas, (pad, output_size // 2), (output_size - pad, output_size // 2), cv_gray, 1)
    cv2.line(canvas, (output_size // 2, pad), (output_size // 2, output_size - pad), cv_gray, 1)

    for contour in norm_collection.contours:
        if len(contour.points) < 2:
            continue

        pts = np.array(
            [[_norm_to_canvas(p, half, draw_range, pad)] for p in contour.points],
            dtype=np.int32,
        )
        color = (0, 220, 80) if contour.is_external else (80, 80, 220)
        import cv2
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=1)

    print(f"[normalization] Preview rendered ({output_size}x{output_size}px)")
    return canvas


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize_point(p: Point, cx: float, cy: float, scale: float) -> Point:
    return Point(
        x = (p.x - cx) / scale,
        y = -(p.y - cy) / scale,   # flip y axis
    )


def _norm_to_canvas(
    p: Point,
    half: float,
    draw_range: float,
    pad: int,
) -> Tuple[int, int]:
    """Map a normalized [-1,1] point to canvas pixel coordinates."""
    px = int(pad + (p.x + 1.0) / 2.0 * draw_range)
    py = int(pad + (1.0 - (p.y + 1.0) / 2.0) * draw_range)  # flip y back for screen
    return (px, py)
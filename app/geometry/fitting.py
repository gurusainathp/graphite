"""
Graphite — fitting.py
Responsibility: Detect geometric primitives from simplified contours.
  - Line segments: consecutive collinear point runs
  - Curve segments: non-linear point sequences (sampled for future spline fitting)
  - Circle detection: via minimum enclosing circle heuristic
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from geometry.models import Point, Contour, ContourCollection


# ── Primitive models ──────────────────────────────────────────────────────────

@dataclass
class LineSegment:
    """A straight line segment between two points with slope/intercept."""
    p1: Point
    p2: Point

    @property
    def dx(self) -> float:
        return self.p2.x - self.p1.x

    @property
    def dy(self) -> float:
        return self.p2.y - self.p1.y

    @property
    def is_vertical(self) -> bool:
        return abs(self.dx) < 1e-9

    @property
    def slope(self) -> float | None:
        return None if self.is_vertical else self.dy / self.dx

    @property
    def intercept(self) -> float | None:
        if self.is_vertical:
            return None
        return self.p1.y - self.slope * self.p1.x

    @property
    def length(self) -> float:
        return float(np.hypot(self.dx, self.dy))

    @property
    def x_min(self) -> float:
        return min(self.p1.x, self.p2.x)

    @property
    def x_max(self) -> float:
        return max(self.p1.x, self.p2.x)

    def __repr__(self) -> str:
        if self.is_vertical:
            return f"LineSegment(vertical x={self.p1.x:.2f}, len={self.length:.1f})"
        return f"LineSegment(m={self.slope:.3f}, b={self.intercept:.3f}, x=[{self.x_min:.2f},{self.x_max:.2f}])"


@dataclass
class CurveSegment:
    """A non-linear sequence of points — preserved for future spline fitting."""
    points: List[Point]

    @property
    def point_count(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"CurveSegment(points={self.point_count})"


@dataclass
class CirclePrimitive:
    """A detected circle primitive."""
    center: Point
    radius: float

    @property
    def circularity_domain(self) -> Tuple[float, float]:
        return (self.center.x - self.radius, self.center.x + self.radius)

    def __repr__(self) -> str:
        return f"CirclePrimitive(center={self.center}, r={self.radius:.2f})"


@dataclass
class FittingResult:
    """All primitives detected from a single contour."""
    lines:  List[LineSegment]  = field(default_factory=list)
    curves: List[CurveSegment] = field(default_factory=list)
    circle: CirclePrimitive | None = None

    @property
    def total_primitives(self) -> int:
        n = len(self.lines) + len(self.curves)
        return n + (1 if self.circle else 0)


# ── Main fitting functions ────────────────────────────────────────────────────

# Threshold: residual (px) below which a run of points is called "linear"
LINE_RESIDUAL_THRESHOLD = 2.5
# Circularity: ratio of area to bounding circle — near 1.0 = circle
CIRCULARITY_THRESHOLD   = 0.80
# Minimum segment length to bother emitting
MIN_SEGMENT_LENGTH      = 3.0


def fit_contour(contour: Contour) -> FittingResult:
    """
    Classify the points of a single contour into line segments, curve segments,
    and optionally a circle primitive.

    Strategy:
        1. Check for circle via area/enclosing-circle ratio.
        2. Walk consecutive point triples; accumulate a "line run" while
           perpendicular residual stays below threshold.
        3. When residual spikes, flush the run as a LineSegment or CurveSegment.

    Args:
        contour: Simplified Contour in pixel space.

    Returns:
        FittingResult with detected primitives.
    """
    result = FittingResult()
    pts = contour.points

    if len(pts) < 2:
        return result

    # ── Circle check ──────────────────────────────────────────────────────────
    raw = np.array([[int(p.x), int(p.y)] for p in pts], dtype=np.float32)
    (cx, cy), radius = cv2.minEnclosingCircle(raw.reshape(-1, 1, 2))
    enclosing_area = np.pi * radius ** 2
    if enclosing_area > 0:
        circularity = contour.area / enclosing_area
        if circularity >= CIRCULARITY_THRESHOLD and len(pts) >= 8:
            result.circle = CirclePrimitive(
                center=Point(float(cx), float(cy)),
                radius=float(radius),
            )
            return result  # treat the whole contour as a circle

    # ── Line / curve segmentation ─────────────────────────────────────────────
    run: List[Point] = [pts[0]]

    for i in range(1, len(pts)):
        run.append(pts[i])
        if len(run) < 3:
            continue

        residual = _max_perpendicular_residual(run)
        if residual > LINE_RESIDUAL_THRESHOLD:
            # Flush all but the last point as a segment
            _flush_run(run[:-1], result)
            run = [run[-2], run[-1]]   # keep overlap for continuity

    _flush_run(run, result)  # flush remainder
    return result


def fit_collection(collection: ContourCollection) -> List[FittingResult]:
    """
    Fit primitives for every contour in a collection.

    Returns:
        List of FittingResult, one per contour.
    """
    results = []
    total_lines  = 0
    total_curves = 0
    total_circles = 0

    for contour in collection.contours:
        r = fit_contour(contour)
        results.append(r)
        total_lines   += len(r.lines)
        total_curves  += len(r.curves)
        total_circles += 1 if r.circle else 0

    print(f"[fitting] Contours fit : {len(results)}")
    print(f"[fitting] Line segs   : {total_lines}")
    print(f"[fitting] Curve segs  : {total_curves}")
    print(f"[fitting] Circles     : {total_circles}")

    return results


def draw_segments(
    results: List[FittingResult],
    width: int,
    height: int,
) -> np.ndarray:
    """Render detected primitives onto a black canvas."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for r in results:
        # Lines → green
        for seg in r.lines:
            p1 = (int(seg.p1.x), int(seg.p1.y))
            p2 = (int(seg.p2.x), int(seg.p2.y))
            cv2.line(canvas, p1, p2, (0, 220, 80), 1)
            cv2.circle(canvas, p1, 2, (0, 180, 60), -1)
            cv2.circle(canvas, p2, 2, (0, 180, 60), -1)

        # Curves → blue
        for seg in r.curves:
            pts_arr = np.array(
                [[[int(p.x), int(p.y)]] for p in seg.points], dtype=np.int32
            )
            cv2.polylines(canvas, [pts_arr], isClosed=False, color=(80, 120, 255), thickness=1)

        # Circles → orange
        if r.circle:
            c = r.circle
            cv2.circle(canvas, (int(c.center.x), int(c.center.y)), int(c.radius), (0, 165, 255), 1)

    print(f"[fitting] Rendered segment visualization")
    return canvas


# ── Internal helpers ──────────────────────────────────────────────────────────

def _max_perpendicular_residual(run: List[Point]) -> float:
    """
    Fit a line through the first and last point of `run`; return the maximum
    perpendicular distance of all interior points from that line.
    """
    if len(run) < 3:
        return 0.0

    x0, y0 = run[0].x,  run[0].y
    x1, y1 = run[-1].x, run[-1].y
    dx, dy  = x1 - x0, y1 - y0
    length  = np.hypot(dx, dy)

    if length < 1e-9:
        return 0.0

    max_dist = 0.0
    for p in run[1:-1]:
        dist = abs(dy * p.x - dx * p.y + x1 * y0 - y1 * x0) / length
        if dist > max_dist:
            max_dist = dist

    return max_dist


def _flush_run(run: List[Point], result: FittingResult) -> None:
    """Decide whether a run is a line or a curve and append to result."""
    if len(run) < 2:
        return

    seg = LineSegment(run[0], run[-1])

    if seg.length < MIN_SEGMENT_LENGTH:
        return

    residual = _max_perpendicular_residual(run)
    if residual <= LINE_RESIDUAL_THRESHOLD:
        result.lines.append(seg)
    else:
        result.curves.append(CurveSegment(points=list(run)))
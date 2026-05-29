"""
Graphite — equations.py
Responsibility: Convert geometric primitives into Desmos-compatible equation strings.

Supported primitives:
    LineSegment   →  y = mx + b {x1 < x < x2}
                     x = c      {y1 < y < y2}   (vertical)
    CirclePrimitive → (x - h)^2 + (y - k)^2 = r^2
    CurveSegment  →  parametric point list  (x1, y1), (x2, y2), ...
"""

from dataclasses import dataclass, field
from typing import List

from geometry.fitting import LineSegment, CurveSegment, CirclePrimitive, FittingResult


# ── Equation models ───────────────────────────────────────────────────────────

@dataclass
class DesmosExpression:
    """A single Desmos-compatible expression string with its source primitive type."""
    expression: str
    kind: str          # 'line' | 'vertical' | 'circle' | 'curve_points'

    def __repr__(self) -> str:
        return f"[{self.kind}] {self.expression}"


@dataclass
class EquationSet:
    """All equations generated from one image."""
    expressions: List[DesmosExpression] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.expressions)

    def by_kind(self, kind: str) -> List[DesmosExpression]:
        return [e for e in self.expressions if e.kind == kind]

    def print_stats(self) -> None:
        kinds = {}
        for e in self.expressions:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        print(f"[equations] Total expressions : {self.count}")
        for k, v in sorted(kinds.items()):
            print(f"[equations]   {k:<15}: {v}")


# ── Generation ────────────────────────────────────────────────────────────────

DOMAIN_PADDING = 0.0   # extra padding on x-domain clamps (in coordinate units)


def line_to_expression(seg: LineSegment) -> DesmosExpression:
    """
    Convert a LineSegment to a Desmos y = mx + b {domain} expression.

    Vertical lines become x = c {y1 < y < y2}.
    """
    if seg.is_vertical:
        x_val = round(seg.p1.x, 4)
        y_min = round(min(seg.p1.y, seg.p2.y), 4)
        y_max = round(max(seg.p1.y, seg.p2.y), 4)
        expr  = f"x = {x_val} {{{y_min} < y < {y_max}}}"
        return DesmosExpression(expression=expr, kind="vertical")

    m = seg.slope
    b = seg.intercept
    x1 = round(seg.x_min - DOMAIN_PADDING, 4)
    x2 = round(seg.x_max + DOMAIN_PADDING, 4)

    # Build slope term
    if abs(m) < 1e-9:
        slope_str = ""          # horizontal: just b
    elif abs(m - 1.0) < 1e-9:
        slope_str = "x"
    elif abs(m + 1.0) < 1e-9:
        slope_str = "-x"
    else:
        slope_str = f"{round(m, 4)}x"

    # Build intercept term
    if slope_str == "":
        body = str(round(b, 4))
    elif abs(b) < 1e-9:
        body = slope_str
    elif b > 0:
        body = f"{slope_str} + {round(b, 4)}"
    else:
        body = f"{slope_str} - {round(abs(b), 4)}"

    expr = f"y = {body} {{{x1} < x < {x2}}}"
    return DesmosExpression(expression=expr, kind="line")


def circle_to_expression(circle: CirclePrimitive) -> DesmosExpression:
    """
    Convert a CirclePrimitive to a Desmos (x-h)^2 + (y-k)^2 = r^2 expression.
    """
    h = round(circle.center.x, 4)
    k = round(circle.center.y, 4)
    r = round(circle.radius,   4)

    h_str = f"(x - {h})^2" if h >= 0 else f"(x + {abs(h)})^2"
    k_str = f"(y - {k})^2" if k >= 0 else f"(y + {abs(k)})^2"

    expr = f"{h_str} + {k_str} = {r}^2"
    return DesmosExpression(expression=expr, kind="circle")


def curve_to_expression(curve: CurveSegment) -> DesmosExpression:
    """
    Represent a CurveSegment as a Desmos point list.
    Desmos can plot raw (x, y) pairs; useful until full spline fitting is added.

    Format: (x1, y1), (x2, y2), ...
    Points are sampled to keep Desmos manageable (max 50 pts).
    """
    pts = curve.points
    step = max(1, len(pts) // 50)
    sampled = pts[::step]

    pairs = ", ".join(f"({round(p.x, 3)}, {round(p.y, 3)})" for p in sampled)
    return DesmosExpression(expression=pairs, kind="curve_points")


def generate_equations(results: List[FittingResult]) -> EquationSet:
    """
    Generate a full EquationSet from a list of FittingResults.

    Args:
        results: One FittingResult per contour (from fit_collection).

    Returns:
        EquationSet with all expressions ready for export.
    """
    eq_set = EquationSet()

    for r in results:
        if r.circle:
            eq_set.expressions.append(circle_to_expression(r.circle))
            continue   # circle subsumes line/curve for that contour

        for seg in r.lines:
            eq_set.expressions.append(line_to_expression(seg))

        for curve in r.curves:
            eq_set.expressions.append(curve_to_expression(curve))

    eq_set.print_stats()
    return eq_set
"""
Graphite — filtering.py
Responsibility: Remove noise contours and useless fragments from a ContourCollection.
"""

from geometry.models import Contour, ContourCollection


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MIN_AREA      = 20.0    # px²  — drop specks smaller than this
DEFAULT_MIN_PERIMETER = 10.0    # px   — drop contours with negligible boundary
DEFAULT_MIN_POINTS    = 5       # pts  — drop contours with too few points to form a curve


def filter_contours(
    collection: ContourCollection,
    min_area:      float = DEFAULT_MIN_AREA,
    min_perimeter: float = DEFAULT_MIN_PERIMETER,
    min_points:    int   = DEFAULT_MIN_POINTS,
) -> ContourCollection:
    """
    Return a new ContourCollection with noise and fragment contours removed.

    Filtering criteria (all must pass):
        - area      >= min_area
        - perimeter >= min_perimeter
        - point count >= min_points

    Args:
        collection:     Source ContourCollection (not mutated).
        min_area:       Minimum enclosed area in px². Default: 20.
        min_perimeter:  Minimum arc length in px.   Default: 10.
        min_points:     Minimum number of points.   Default: 5.

    Returns:
        New ContourCollection containing only the surviving contours.
    """
    before = collection.count
    kept: list[Contour] = []
    rejected_counts = {"area": 0, "perimeter": 0, "points": 0}

    for c in collection.contours:
        if c.area < min_area:
            rejected_counts["area"] += 1
            continue
        if c.perimeter < min_perimeter:
            rejected_counts["perimeter"] += 1
            continue
        if c.point_count < min_points:
            rejected_counts["points"] += 1
            continue
        kept.append(c)

    filtered = ContourCollection(
        contours     = kept,
        image_width  = collection.image_width,
        image_height = collection.image_height,
    )

    after    = filtered.count
    removed  = before - after

    print(f"[filtering] Before : {before} contours")
    print(f"[filtering] Removed: {removed} "
          f"(area={rejected_counts['area']}, "
          f"perimeter={rejected_counts['perimeter']}, "
          f"points={rejected_counts['points']})")
    print(f"[filtering] After  : {after} contours "
          f"({len(filtered.external)} external, {len(filtered.internal)} internal)")

    return filtered
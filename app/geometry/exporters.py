"""
Graphite — exporters.py
Responsibility: Write pipeline results to disk in multiple formats.

Outputs:
    desmos_export.txt  — one expression per line, paste directly into Desmos
    equations.txt      — annotated listing with kind labels
    geometry.json      — full structured export for downstream tooling
"""

import json
from pathlib import Path
from typing import List

from geometry.equations import EquationSet, DesmosExpression
from geometry.fitting   import FittingResult, LineSegment, CurveSegment, CirclePrimitive


# ── Public exporters ──────────────────────────────────────────────────────────

def export_desmos(eq_set: EquationSet, output_dir: Path) -> Path:
    """
    Write a plain-text file with one Desmos expression per line.
    Paste the contents directly into the Desmos expression list.

    Returns the path of the written file.
    """
    out_path = output_dir / "desmos_export.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [e.expression for e in eq_set.expressions]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[exporters] desmos_export.txt  — {len(lines)} expressions → {out_path}")
    return out_path


def export_annotated(eq_set: EquationSet, output_dir: Path) -> Path:
    """
    Write an annotated equations file with kind labels for human review.

    Returns the path of the written file.
    """
    out_path = output_dir / "equations.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Graphite — Generated Equations\n")
        f.write(f"# Total: {eq_set.count}\n\n")

        for kind in ("line", "vertical", "circle", "curve_points"):
            group = eq_set.by_kind(kind)
            if not group:
                continue
            f.write(f"# ── {kind.upper()} ({len(group)}) ───────────────────\n")
            for expr in group:
                f.write(f"{expr.expression}\n")
            f.write("\n")

    print(f"[exporters] equations.txt      — {eq_set.count} expressions → {out_path}")
    return out_path


def export_geometry_json(
    results: List[FittingResult],
    output_dir: Path,
) -> Path:
    """
    Write a structured JSON file with all detected primitives.
    Useful for debugging and future pipeline stages.

    Returns the path of the written file.
    """
    out_path = output_dir / "geometry.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {"contours": []}

    for idx, r in enumerate(results):
        entry = {
            "contour_index": idx,
            "circle":  None,
            "lines":   [],
            "curves":  [],
        }

        if r.circle:
            c = r.circle
            entry["circle"] = {
                "center": {"x": round(c.center.x, 4), "y": round(c.center.y, 4)},
                "radius": round(c.radius, 4),
            }
        else:
            for seg in r.lines:
                entry["lines"].append({
                    "p1":        {"x": round(seg.p1.x, 4), "y": round(seg.p1.y, 4)},
                    "p2":        {"x": round(seg.p2.x, 4), "y": round(seg.p2.y, 4)},
                    "slope":     round(seg.slope, 6) if seg.slope is not None else None,
                    "intercept": round(seg.intercept, 6) if seg.intercept is not None else None,
                    "length":    round(seg.length, 3),
                    "vertical":  seg.is_vertical,
                })

            for curve in r.curves:
                entry["curves"].append({
                    "point_count": curve.point_count,
                    "points": [
                        {"x": round(p.x, 4), "y": round(p.y, 4)}
                        for p in curve.points
                    ],
                })

        payload["contours"].append(entry)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    total_contours = len(results)
    print(f"[exporters] geometry.json      — {total_contours} contours → {out_path}")
    return out_path


def export_all(
    eq_set: EquationSet,
    results: List[FittingResult],
    output_dir: Path,
) -> None:
    """Convenience wrapper — runs all three exporters in one call."""
    print("[exporters] Writing all outputs...")
    export_desmos(eq_set, output_dir)
    export_annotated(eq_set, output_dir)
    export_geometry_json(results, output_dir)
    print("[exporters] Done.")
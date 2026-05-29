"""
Graphite - Image to Desmos Graph Converter
main.py: Pipeline entry point — orchestrates all processing stages.
"""

from pathlib import Path
import cv2

# ── Phase 2: Image Processing ─────────────────────────────────────────────────
from image_processing.loader         import load_image
from image_processing.preprocessing  import convert_to_grayscale, apply_gaussian_blur, resize_image
from image_processing.edge_detection import detect_edges

# ── Phase 3: Geometry ─────────────────────────────────────────────────────────
from geometry.contours      import extract_contours, draw_contours
from geometry.filtering     import filter_contours
from geometry.normalization import normalize_collection, render_normalized_preview

# ── Phase 4: Simplification + Equations ──────────────────────────────────────
from geometry.simplification import simplify_collection, draw_simplified
from geometry.fitting        import fit_collection, draw_segments
from geometry.equations      import generate_equations
from geometry.exporters      import export_all


# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "assets" / "input" / "sample.png"
OUTPUT_DIR = BASE_DIR / "assets" / "output"


# ── Output Utility ────────────────────────────────────────────────────────────

def save_image(array, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    cv2.imwrite(str(out_path), array)
    print(f"[main] Saved → {out_path}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Graphite Pipeline ─────────────────────────────")

    # ── Phase 2: Image Preprocessing ──────────────────────────────────────────
    print("\n[Phase 2] Image Preprocessing")
    image   = load_image(INPUT_PATH)
    image   = resize_image(image, max_dim=1024)
    gray    = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(gray, kernel_size=5)
    edges   = detect_edges(blurred, threshold_low=50, threshold_high=150)

    save_image(gray,    "grayscale.png")
    save_image(blurred, "blurred.png")
    save_image(edges,   "edges.png")

    # ── Phase 3: Contour Extraction & Geometry ────────────────────────────────
    print("\n[Phase 3] Contour Extraction")
    collection = extract_contours(edges)
    collection.print_stats()
    save_image(draw_contours(collection), "contours.png")

    filtered = filter_contours(collection, min_area=20, min_perimeter=10, min_points=5)
    save_image(draw_contours(filtered),   "filtered_contours.png")

    normalized = normalize_collection(filtered)
    save_image(render_normalized_preview(normalized), "normalized_preview.png")

    # ── Phase 4: Simplification + Equation Generation ─────────────────────────
    print("\n[Phase 4] Curve Simplification & Equation Generation")
    simplified = simplify_collection(filtered, epsilon_ratio=0.01)
    save_image(draw_simplified(simplified), "simplified_contours.png")

    fit_results = fit_collection(simplified)
    save_image(
        draw_segments(fit_results, collection.image_width, collection.image_height),
        "line_segments.png",
    )

    eq_set = generate_equations(fit_results)
    export_all(eq_set, fit_results, OUTPUT_DIR)

    print("\n── Pipeline complete ─────────────────────────────\n")


if __name__ == "__main__":
    main()
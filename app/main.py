"""
Graphite - Image to Desmos Graph Converter
main.py: Pipeline entry point — orchestrates all processing stages.
"""

from pathlib import Path

import cv2

from image_processing.loader        import load_image
from image_processing.preprocessing import convert_to_grayscale, apply_gaussian_blur, resize_image
from image_processing.edge_detection import detect_edges

from geometry.contours       import extract_contours, draw_contours
from geometry.filtering      import filter_contours
from geometry.normalization  import normalize_collection, render_normalized_preview


# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "assets" / "input" / "sample.png"
OUTPUT_DIR = BASE_DIR / "assets" / "output"


# ── Output Utility ─────────────────────────────────────────────────────────────

def save_image(array, filename: str) -> None:
    """Save a NumPy array as an image to assets/output/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    cv2.imwrite(str(out_path), array)
    print(f"[main] Saved → {out_path}")


# ── Pipeline ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Graphite Pipeline ─────────────────────────────")

    # ── Phase 2: Image Preprocessing ──────────────────
    image   = load_image(INPUT_PATH)
    image   = resize_image(image, max_dim=1024)
    gray    = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(gray, kernel_size=5)
    edges   = detect_edges(blurred, threshold_low=50, threshold_high=150)

    save_image(gray,    "grayscale.png")
    save_image(blurred, "blurred.png")
    save_image(edges,   "edges.png")

    # ── Phase 3: Contour Extraction & Geometry ─────────
    collection = extract_contours(edges)
    collection.print_stats()

    contour_vis = draw_contours(collection, label="all")
    save_image(contour_vis, "contours.png")

    filtered = filter_contours(collection, min_area=20, min_perimeter=10, min_points=5)
    filtered_vis = draw_contours(filtered, label="all")
    save_image(filtered_vis, "filtered_contours.png")

    normalized   = normalize_collection(filtered)
    norm_preview = render_normalized_preview(normalized, output_size=800)
    save_image(norm_preview, "normalized_preview.png")

    print("\n── Pipeline complete ─────────────────────────────\n")


if __name__ == "__main__":
    main()
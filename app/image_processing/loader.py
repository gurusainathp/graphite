"""
Graphite — loader.py
Responsibility: Load, validate, and return image with basic metadata.
"""

from pathlib import Path
from PIL import Image


SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_image(path: str | Path) -> Image.Image:
    """
    Load an image from the given path.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"[loader] Image not found: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"[loader] Unsupported format '{image_path.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    image = Image.open(image_path).convert("RGB")

    _print_metadata(image_path, image)

    return image


def _print_metadata(path: Path, image: Image.Image) -> None:
    """Print basic image metadata to stdout."""
    print(f"[loader] File     : {path.name}")
    print(f"[loader] Size     : {image.width} x {image.height} px")
    print(f"[loader] Mode     : {image.mode}")
    print(f"[loader] Format   : {path.suffix.upper().lstrip('.')}")
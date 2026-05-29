"""
Graphite — preprocessing.py
Responsibility: Grayscale conversion, resizing, and noise reduction via Gaussian blur.
"""

import cv2
import numpy as np
from PIL import Image


def convert_to_grayscale(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL RGB image to a grayscale NumPy array.

    Args:
        image: PIL Image in RGB mode.

    Returns:
        2D NumPy array (uint8) representing the grayscale image.
    """
    rgb_array = np.array(image)
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    print(f"[preprocessing] Grayscale conversion done — shape: {gray.shape}")
    return gray


def apply_gaussian_blur(
    gray: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0,
) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise before edge detection.

    Args:
        gray:        2D grayscale NumPy array.
        kernel_size: Size of the Gaussian kernel (must be odd). Default: 5.
        sigma:       Gaussian standard deviation. 0 = auto-calculated by OpenCV.

    Returns:
        Blurred 2D NumPy array (uint8).
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"[preprocessing] kernel_size must be odd, got {kernel_size}.")

    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    print(f"[preprocessing] Gaussian blur applied — kernel: {kernel_size}x{kernel_size}, sigma: {sigma or 'auto'}")
    return blurred


def resize_image(image: Image.Image, max_dim: int = 1024) -> Image.Image:
    """
    Proportionally resize an image so its longest side is at most max_dim.
    No-op if the image is already within bounds.

    Args:
        image:   PIL Image.
        max_dim: Maximum allowed size for width or height. Default: 1024.

    Returns:
        Resized (or unchanged) PIL Image.
    """
    w, h = image.size
    if max(w, h) <= max_dim:
        return image

    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    resized = image.resize(new_size, Image.LANCZOS)
    print(f"[preprocessing] Resized: {w}x{h} → {new_size[0]}x{new_size[1]}")
    return resized
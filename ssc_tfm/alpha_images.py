"""Convert alpha-f images between detector and display row orientations.

Detector arrays place alpha_min at the top. Display PNGs place alpha_max at
the top, so the cyclic-frequency axis increases upward when viewed normally.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

DETECTOR_ORIENTATION = "alpha_min_at_top"
DISPLAY_ORIENTATION = "alpha_max_at_top"


def _gray_u8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2 or array.dtype != np.uint8 or not array.size:
        raise ValueError("An alpha-f image must be a nonempty two-dimensional uint8 array.")
    return array


def detector_to_display(image: np.ndarray) -> np.ndarray:
    """Return a contiguous display array, with the largest alpha at row zero."""
    return np.ascontiguousarray(np.flipud(_gray_u8(image)))


def display_to_detector(image: np.ndarray) -> np.ndarray:
    """Restore the detector's smallest-alpha-at-row-zero array exactly."""
    return np.ascontiguousarray(np.flipud(_gray_u8(image)))


def read_alpha_f_image(path: str | Path, orientation: str | None = None) -> np.ndarray:
    """Load a PNG as a detector array.

    An explicit orientation comes from the record's metadata. Without one,
    use the PNG orientation tag, falling back to the legacy detector layout.
    """
    with Image.open(path) as source:
        stored_orientation = orientation or source.info.get("alpha_f_image_orientation", DETECTOR_ORIENTATION)
        pixels = np.asarray(source.convert("L"), dtype=np.uint8)
    if stored_orientation == DETECTOR_ORIENTATION:
        return np.ascontiguousarray(pixels)
    if stored_orientation == DISPLAY_ORIENTATION:
        return display_to_detector(pixels)
    raise ValueError(f"Unsupported alpha-f image orientation: {stored_orientation}")


def save_alpha_f_image(image: np.ndarray, path: str | Path,
                       alpha_min_hz: float | None = None,
                       alpha_max_hz: float | None = None) -> None:
    """Save a detector array as an upward-increasing-alpha display PNG."""
    metadata = PngInfo()
    metadata.add_text("alpha_f_image_orientation", DISPLAY_ORIENTATION)
    if alpha_min_hz is not None:
        metadata.add_text("alpha_min_hz", str(float(alpha_min_hz)))
    if alpha_max_hz is not None:
        metadata.add_text("alpha_max_hz", str(float(alpha_max_hz)))
    Image.fromarray(detector_to_display(image)).save(path, pnginfo=metadata)

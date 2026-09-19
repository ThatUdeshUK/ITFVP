"""TIFF -> QPixmap preview helper for the Images page and thumbnails.

Shows frames as their original channel layout (grayscale stays grayscale,
multi-channel stays multi-channel) rather than collapsing channels the way
the pipeline's own to_gray() does for PIV/traction computation — that
collapse is a computation detail (see stabilize.py/piv.py), not how the
image should look when just browsing it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap


def _normalize_uint8(arr: np.ndarray) -> np.ndarray:
    """Percentile auto-contrast stretch to 8-bit — needed because raw
    microscopy TIFFs are often 16-bit with signal packed into a narrow
    range that would otherwise render as near-black."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255).astype(np.uint8)


def load_preview_qimage(path: Path, max_size: int = 220) -> QImage:
    """Safe to call off the GUI thread (unlike QPixmap) — used to build
    thumbnails on a worker thread; convert to QPixmap on the main thread."""
    arr = tifffile.imread(str(path))

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        channels = arr.shape[-1]
        rgb = np.ascontiguousarray(
            np.stack([_normalize_uint8(arr[..., c]) for c in range(channels)], axis=-1)
        )
        h, w, _ = rgb.shape
        fmt = QImage.Format.Format_RGBA8888 if channels == 4 else QImage.Format.Format_RGB888
        image = QImage(rgb.data, w, h, w * channels, fmt).copy()
    else:
        # Single-channel, or an uncommon channel count we can't map to
        # RGB(A) — show the first plane as a plain grayscale image.
        plane = arr[..., 0] if arr.ndim == 3 else arr
        gray = np.ascontiguousarray(_normalize_uint8(plane))
        h, w = gray.shape
        image = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8).copy()

    return image.scaled(
        max_size, max_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def load_preview_pixmap(path: Path, max_size: int = 220) -> QPixmap:
    return QPixmap.fromImage(load_preview_qimage(path, max_size))

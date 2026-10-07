"""CFAR ship detection.

A CFAR detector compares each pixel with a local estimate of the clutter level taken from a
ring of training cells around it, outside a guard window that keeps the ship itself out of
the estimate. Detection = ratio image > threshold.

- CA-CFAR estimates the clutter by the *mean* of the ring (optimal in homogeneous clutter,
  but a second ship or a bright edge inside the ring inflates it: target masking, clutter edges).
- OS-CFAR uses an *ordered statistic* (a high percentile) of the ring, which ignores a minority
  of contaminated cells.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes, center_of_mass, gaussian_filter, label, percentile_filter,
                           uniform_filter)

GUARD_HALF, OUTER_HALF = 7, 15  # 15x15 guard window, 31x31 outer window


def ring_footprint(guard_half: int = GUARD_HALF, outer_half: int = OUTER_HALF) -> np.ndarray:
    fp = np.ones((2 * outer_half + 1,) * 2, bool)
    c = outer_half
    fp[c - guard_half : c + guard_half + 1, c - guard_half : c + guard_half + 1] = False
    return fp


def ca_background(img: np.ndarray) -> np.ndarray:
    outer, inner = 2 * OUTER_HALF + 1, 2 * GUARD_HALF + 1
    total = uniform_filter(img, outer, mode="reflect") * outer**2
    hole = uniform_filter(img, inner, mode="reflect") * inner**2
    return (total - hole) / (outer**2 - inner**2)


def os_background(img: np.ndarray, percentile: float = 75.0) -> np.ndarray:
    return percentile_filter(img, percentile, footprint=ring_footprint(), mode="reflect")


def ratio_map(img: np.ndarray, method: str) -> np.ndarray:
    bg = ca_background(img) if method == "CA" else os_background(img)
    return img / np.maximum(bg, 1e-6)


def land_mask(img: np.ndarray, factor: float = 2.0, margin: int = 9, min_area: int = 1200) -> np.ndarray:
    """Coarse land mask: heavily smoothed intensity far above the scene median, keeping only
    *large* connected regions (land is huge, even a very bright ship is not), then dilated so
    the bright coastline edge is excluded too."""
    bright = gaussian_filter(img, 6) > factor * np.median(img)
    labels, n = label(bright)
    if n == 0:
        return np.zeros_like(bright)
    sizes = np.bincount(labels.ravel())
    big = np.isin(labels, np.flatnonzero(sizes >= min_area)[np.flatnonzero(sizes >= min_area) > 0])
    if not big.any():
        return big
    big = binary_fill_holes(binary_closing(big, iterations=8))  # dark land texture must not leave holes
    return binary_dilation(big, iterations=margin)


def detect(ratio: np.ndarray, threshold: float, exclude: np.ndarray | None = None,
           min_area: int = 3, merge: int = 2) -> np.ndarray:
    """Threshold, merge nearby pixels into one object, drop tiny ones. Returns (N, 2) of (x, y)."""
    mask = ratio > threshold
    if exclude is not None:
        mask &= ~exclude
    if not mask.any():
        return np.empty((0, 2))
    labels, n = label(binary_dilation(mask, iterations=merge))
    idx = np.arange(1, n + 1)
    areas = np.array([(mask & (labels == i)).sum() for i in idx])
    keep = idx[areas >= min_area]
    if len(keep) == 0:
        return np.empty((0, 2))
    return np.array([(x, y) for y, x in center_of_mass(mask, labels, keep)])

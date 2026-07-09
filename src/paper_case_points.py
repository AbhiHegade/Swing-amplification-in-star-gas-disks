"""Shared disk-case markers and styling for paper figures."""

import numpy as np

# Thin-disk cases in (Qs, Qg) order used across notebooks:
# index 0: red diamond  (Qs=5,    Qg=1.43)
# index 1: red star    (Qs=2.2, Qg=2.2)
# index 2: red circle  (Qs=1.43, Qg=5)
POINTS_THIN_DISKS = np.array(
    [
        [1.0 / 0.2, 1.0 / 0.7],
        [2.2, 2.2],
        [1.0 / 0.7, 1.0 / 0.2],
    ]
)

POINTS_THICK_DISKS = np.array(
    [
        [1.35, 1.35],
    ]
)

THIN_DISK_MARKERS = ["D", "*", "o"]
THICK_DISK_MARKERS = ["*"]

SCATTER_BASE_S = 50
MARKER_SCALE = {
    "D": 0.85,
    "*": 1.15,
    "o": 1.0,
}


def scatter_marker_size(marker: str, base_s: float = SCATTER_BASE_S) -> float:
    scale = MARKER_SCALE.get(marker, 1.0)
    return base_s * scale * scale


def line_marker_size(marker: str, base_size: float) -> float:
    scale = MARKER_SCALE.get(marker, 1.0)
    return base_size * scale

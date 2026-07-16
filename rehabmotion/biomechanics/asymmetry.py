from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def calculate_asymmetry(
    left_angles: Sequence[float], right_angles: Sequence[float]
) -> np.ndarray:
    """Return absolute left-right angular difference for aligned frames."""
    left = np.asarray(left_angles, dtype=float)
    right = np.asarray(right_angles, dtype=float)
    if left.shape != right.shape:
        raise ValueError("left and right angle series must have the same shape")
    return np.abs(left - right)

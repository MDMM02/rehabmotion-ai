from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def calculate_angle(
    point_a: Sequence[float],
    point_b: Sequence[float],
    point_c: Sequence[float],
) -> float:
    """Calculate angle ABC in degrees, with point B as the joint center."""
    a = np.asarray(point_a, dtype=float)[:2]
    b = np.asarray(point_b, dtype=float)[:2]
    c = np.asarray(point_c, dtype=float)[:2]
    if a.shape != (2,) or b.shape != (2,) or c.shape != (2,):
        raise ValueError("Each point must contain at least x and y coordinates.")
    if not np.all(np.isfinite(np.concatenate((a, b, c)))):
        return float("nan")

    vector_ba = a - b
    vector_bc = c - b
    denominator = np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc)
    if denominator <= np.finfo(float).eps:
        return float("nan")

    cosine = np.clip(np.dot(vector_ba, vector_bc) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def calculate_range_of_motion(angle_series: Sequence[float]) -> float:
    """Return max minus min for finite joint-angle values."""
    values = np.asarray(angle_series, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite) - np.min(finite))

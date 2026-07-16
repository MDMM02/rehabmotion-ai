from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _odd_window(window_size: int) -> int:
    if window_size < 3:
        raise ValueError("window_size must be at least 3")
    return window_size if window_size % 2 else window_size + 1


def smooth_signal(
    signal: Sequence[float],
    method: str = "savgol",
    window_size: int = 7,
    polyorder: int = 2,
    max_gap_frames: int = 2,
) -> np.ndarray:
    """Interpolate short internal gaps, then smooth contiguous valid segments."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if method not in {"savgol", "moving_average"}:
        raise ValueError("method must be 'savgol' or 'moving_average'")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames cannot be negative")

    window = _odd_window(window_size)
    series = pd.Series(values, dtype=float)
    if max_gap_frames:
        series = series.interpolate(
            method="linear",
            limit=max_gap_frames,
            limit_area="inside",
        )
    interpolated = series.to_numpy(dtype=float)

    if method == "moving_average":
        return (
            pd.Series(interpolated)
            .rolling(window=window, center=True, min_periods=1)
            .mean()
            .where(pd.Series(interpolated).notna())
            .to_numpy(dtype=float)
        )

    smoothed = interpolated.copy()
    valid = np.isfinite(interpolated)
    start = 0
    while start < len(interpolated):
        if not valid[start]:
            start += 1
            continue
        end = start
        while end < len(interpolated) and valid[end]:
            end += 1
        segment_length = end - start
        segment_window = min(window, segment_length)
        if segment_window % 2 == 0:
            segment_window -= 1
        if segment_window >= max(3, polyorder + 2):
            smoothed[start:end] = savgol_filter(
                interpolated[start:end],
                window_length=segment_window,
                polyorder=min(polyorder, segment_window - 1),
                mode="interp",
            )
        start = end
    return smoothed

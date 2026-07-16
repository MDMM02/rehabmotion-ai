import numpy as np

from rehabmotion.biomechanics.smoothing import smooth_signal


def test_savgol_smoothing_reduces_single_frame_noise() -> None:
    signal = np.array([100, 101, 99, 130, 101, 100, 99], dtype=float)

    smoothed = smooth_signal(signal, window_size=5)

    assert smoothed.shape == signal.shape
    assert smoothed[3] < signal[3]


def test_short_internal_gap_is_interpolated() -> None:
    smoothed = smooth_signal([100, 101, np.nan, 103, 104], window_size=5)

    assert np.isfinite(smoothed[2])


def test_long_gap_is_preserved() -> None:
    smoothed = smooth_signal(
        [100, 101, np.nan, np.nan, np.nan, 105, 106],
        window_size=5,
        max_gap_frames=2,
    )

    assert np.isnan(smoothed[4])

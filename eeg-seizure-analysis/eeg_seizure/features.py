"""Interpretable time- and frequency-domain EEG features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from .data import SAMPLING_RATE_HZ, signal_columns


EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _band_power(psd: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (frequencies >= low) & (frequencies < high)
    if not mask.any():
        return np.zeros(psd.shape[0])
    frequency_step = float(frequencies[1] - frequencies[0])
    return psd[:, mask].sum(axis=1) * frequency_step


def extract_features(signals: np.ndarray, sampling_rate: float = SAMPLING_RATE_HZ) -> pd.DataFrame:
    """Extract robust handcrafted features from an (n_samples, 178) array."""
    values = np.asarray(signals, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("Les signaux doivent former une matrice 2D.")

    centered = values - values.mean(axis=1, keepdims=True)
    differences = np.diff(values, axis=1)
    q25, median, q75 = np.percentile(values, [25, 50, 75], axis=1)
    frequencies, psd = welch(
        values,
        fs=sampling_rate,
        axis=1,
        nperseg=min(128, values.shape[1]),
        detrend="constant",
    )
    positive_frequency = frequencies > 0
    positive_psd = psd[:, positive_frequency]
    positive_frequencies = frequencies[positive_frequency]
    total_power = positive_psd.sum(axis=1) + np.finfo(float).eps
    probabilities = positive_psd / total_power[:, None]

    features: dict[str, np.ndarray] = {
        "mean": values.mean(axis=1),
        "std": values.std(axis=1),
        "min": values.min(axis=1),
        "max": values.max(axis=1),
        "peak_to_peak": np.ptp(values, axis=1),
        "median": median,
        "iqr": q75 - q25,
        "rms": np.sqrt(np.mean(values**2, axis=1)),
        "mean_absolute_amplitude": np.mean(np.abs(values), axis=1),
        "skewness": skew(values, axis=1, bias=False),
        "kurtosis": kurtosis(values, axis=1, fisher=True, bias=False),
        "zero_crossing_rate": np.mean(centered[:, 1:] * centered[:, :-1] < 0, axis=1),
        "line_length": np.sum(np.abs(differences), axis=1),
        "difference_std": differences.std(axis=1),
        "difference_abs_mean": np.mean(np.abs(differences), axis=1),
        "dominant_frequency_hz": positive_frequencies[np.argmax(positive_psd, axis=1)],
        "spectral_centroid_hz": (positive_psd * positive_frequencies).sum(axis=1) / total_power,
        "spectral_entropy": -(probabilities * np.log2(probabilities + np.finfo(float).eps)).sum(axis=1),
    }

    for band_name, (low, high) in EEG_BANDS.items():
        absolute_power = _band_power(psd, frequencies, low, high)
        features[f"{band_name}_power"] = absolute_power
        features[f"{band_name}_relative_power"] = absolute_power / total_power

    return pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def feature_table(frame: pd.DataFrame, sampling_rate: float = SAMPLING_RATE_HZ) -> pd.DataFrame:
    """Extract features and preserve the metadata needed by modelling."""
    columns = signal_columns(frame)
    features = extract_features(frame[columns].to_numpy(), sampling_rate=sampling_rate)
    metadata = frame[["segment_id", "recording_group", "y", "is_seizure", "class_name"]].reset_index(drop=True)
    return pd.concat([metadata, features], axis=1)


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    metadata = {"segment_id", "recording_group", "y", "is_seizure", "class_name"}
    return [column for column in frame.columns if column not in metadata]


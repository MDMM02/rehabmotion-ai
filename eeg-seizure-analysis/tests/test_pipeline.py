from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_seizure.features import extract_features
from eeg_seizure.modeling import binary_metrics, recording_level_split


def test_feature_extraction_is_finite() -> None:
    rng = np.random.default_rng(42)
    signals = rng.normal(size=(4, 178))
    features = extract_features(signals)
    assert features.shape == (4, 28)
    assert np.isfinite(features.to_numpy()).all()
    assert {"rms", "line_length", "alpha_relative_power", "spectral_entropy"}.issubset(features.columns)


def test_recording_split_has_no_group_leakage() -> None:
    rows = []
    for group_index in range(20):
        target = int(group_index < 5)
        for window in range(3):
            rows.append(
                {
                    "recording_group": f"g{group_index}",
                    "is_seizure": target,
                    "feature": group_index + window,
                }
            )
    features = pd.DataFrame(rows)
    train_mask, test_mask = recording_level_split(features, random_state=7)
    train_groups = set(features.loc[train_mask, "recording_group"])
    test_groups = set(features.loc[test_mask, "recording_group"])
    assert not train_groups & test_groups
    assert train_mask.sum() + test_mask.sum() == len(features)


def test_binary_metrics() -> None:
    metrics = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.3, 0.7, 0.9]))
    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity" if "sensitivity" in metrics else "recall_sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0


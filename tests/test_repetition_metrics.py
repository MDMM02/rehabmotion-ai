import numpy as np
import pandas as pd

from rehabmotion.analysis.movement_metrics import compute_repetition_metrics
from rehabmotion.analysis.repetition_detection import detect_repetitions


def test_per_repetition_metrics_and_regularity() -> None:
    cycle = np.asarray([170, 155, 130, 100, 80, 100, 130, 155, 170], dtype=float)
    knee = np.concatenate((cycle, cycle, cycle))
    timestamps = np.arange(len(knee), dtype=float) * 0.2
    detection = detect_repetitions(knee, timestamps)
    frame_count = len(knee)
    data = pd.DataFrame(
        {
            "timestamp_seconds": timestamps,
            "knee_angle_degrees": knee,
            "left_knee_angle_degrees": np.full(frame_count, np.nan),
            "right_knee_angle_degrees": knee,
            "hip_angle_degrees": knee - 10,
            "left_hip_angle_degrees": np.full(frame_count, np.nan),
            "right_hip_angle_degrees": knee - 10,
            "trunk_lean_degrees": np.linspace(5, 20, frame_count),
            "knee_asymmetry_degrees": np.full(frame_count, np.nan),
        }
    )

    metrics, summary = compute_repetition_metrics(
        data, detection.repetitions, selected_side="right", exercise_type="squat"
    )

    assert len(metrics) == 3
    assert summary.detected_repetitions == 3
    assert summary.tempo_regularity == "high"
    assert summary.mean_knee_rom_degrees == 90.0
    assert "descent_duration_seconds" in metrics.columns

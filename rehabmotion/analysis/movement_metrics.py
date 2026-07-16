from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rehabmotion.biomechanics.angles import calculate_range_of_motion
from rehabmotion.analysis.repetition_detection import RepetitionSegment


@dataclass(frozen=True, slots=True)
class RepetitionSummary:
    detected_repetitions: int
    mean_duration_seconds: float
    duration_cv: float
    tempo_regularity: str
    mean_knee_rom_degrees: float


def _finite_mean(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_max(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else float("nan")


def compute_repetition_metrics(
    kinematics: pd.DataFrame,
    repetitions: tuple[RepetitionSegment, ...] | list[RepetitionSegment],
    selected_side: str,
    exercise_type: str,
) -> tuple[pd.DataFrame, RepetitionSummary]:
    """Compute auditable per-repetition ROM, tempo, trunk and asymmetry metrics."""
    rows: list[dict[str, float | int | str]] = []
    phase_labels = {
        "squat": ("descent", "ascent"),
        "knee-flexion": ("flexion", "extension"),
        "sit-to-stand": ("rise", "sit_down"),
    }
    phase_1_label, phase_2_label = phase_labels[exercise_type]

    for repetition in repetitions:
        segment = kinematics.iloc[
            repetition.start_index : repetition.end_index + 1
        ]
        rows.append(
            {
                "rep": repetition.rep_id,
                "start_time_seconds": repetition.start_time_seconds,
                "turning_time_seconds": repetition.turning_time_seconds,
                "end_time_seconds": repetition.end_time_seconds,
                "duration_seconds": repetition.duration_seconds,
                f"{phase_1_label}_duration_seconds": (
                    repetition.phase_1_duration_seconds
                ),
                f"{phase_2_label}_duration_seconds": (
                    repetition.phase_2_duration_seconds
                ),
                "analyzed_side": selected_side,
                "knee_rom_degrees": calculate_range_of_motion(
                    segment["knee_angle_degrees"]
                ),
                "left_knee_rom_degrees": calculate_range_of_motion(
                    segment["left_knee_angle_degrees"]
                ),
                "right_knee_rom_degrees": calculate_range_of_motion(
                    segment["right_knee_angle_degrees"]
                ),
                "hip_rom_degrees": calculate_range_of_motion(
                    segment["hip_angle_degrees"]
                ),
                "left_hip_rom_degrees": calculate_range_of_motion(
                    segment["left_hip_angle_degrees"]
                ),
                "right_hip_rom_degrees": calculate_range_of_motion(
                    segment["right_hip_angle_degrees"]
                ),
                "max_trunk_lean_degrees": _finite_max(
                    segment["trunk_lean_degrees"]
                ),
                "mean_knee_asymmetry_degrees": _finite_mean(
                    segment["knee_asymmetry_degrees"]
                ),
            }
        )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe, RepetitionSummary(0, float("nan"), float("nan"), "N/A", float("nan"))

    durations = dataframe["duration_seconds"].to_numpy(dtype=float)
    mean_duration = float(np.mean(durations))
    duration_cv = (
        float(np.std(durations, ddof=1) / mean_duration)
        if len(durations) > 1 and mean_duration > 0
        else 0.0
    )
    if duration_cv <= 0.10:
        regularity = "high"
    elif duration_cv <= 0.20:
        regularity = "moderate"
    else:
        regularity = "variable"

    return dataframe, RepetitionSummary(
        detected_repetitions=len(dataframe),
        mean_duration_seconds=mean_duration,
        duration_cv=duration_cv,
        tempo_regularity=regularity,
        mean_knee_rom_degrees=_finite_mean(dataframe["knee_rom_degrees"]),
    )

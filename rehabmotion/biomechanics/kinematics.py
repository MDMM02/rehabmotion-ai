from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rehabmotion.biomechanics.angles import (
    calculate_angle,
    calculate_range_of_motion,
)
from rehabmotion.biomechanics.asymmetry import calculate_asymmetry
from rehabmotion.biomechanics.smoothing import smooth_signal


SIDES = ("left", "right")


@dataclass(frozen=True, slots=True)
class KinematicsMetrics:
    knee_min_degrees: float
    knee_max_degrees: float
    knee_rom_degrees: float
    hip_min_degrees: float
    hip_max_degrees: float
    hip_rom_degrees: float
    trunk_mean_degrees: float
    trunk_max_degrees: float
    valid_frame_rate: float
    knee_asymmetry_mean_degrees: float
    knee_asymmetry_max_degrees: float


@dataclass(slots=True)
class KinematicsResult:
    data: pd.DataFrame
    selected_side: str
    side_visibility: dict[str, float]
    side_reliable_rate: dict[str, float]
    metrics: KinematicsMetrics


def calculate_trunk_lean(
    shoulder: tuple[float, float] | np.ndarray,
    hip: tuple[float, float] | np.ndarray,
) -> float:
    """Calculate unsigned 2D trunk inclination from the image vertical."""
    shoulder_point = np.asarray(shoulder, dtype=float)[:2]
    hip_point = np.asarray(hip, dtype=float)[:2]
    vector = shoulder_point - hip_point
    if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= np.finfo(float).eps:
        return float("nan")
    return float(np.degrees(np.arctan2(abs(vector[0]), abs(vector[1]))))


def _finite_stat(values: pd.Series | np.ndarray, operation: str) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    operations = {
        "min": np.min,
        "max": np.max,
        "mean": np.mean,
    }
    return float(operations[operation](finite))


def _point(
    pivot: pd.DataFrame,
    landmark_name: str,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    return np.column_stack(
        (
            pivot[("x", landmark_name)].to_numpy(dtype=float) * frame_width,
            pivot[("y", landmark_name)].to_numpy(dtype=float) * frame_height,
        )
    )


def _visibility(pivot: pd.DataFrame, landmark_name: str) -> np.ndarray:
    return pivot[("visibility", landmark_name)].to_numpy(dtype=float)


def _angle_series(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
    confidence: np.ndarray,
    min_visibility: float,
) -> np.ndarray:
    values = np.full(len(point_a), np.nan, dtype=float)
    for index in range(len(values)):
        if confidence[index] >= min_visibility:
            values[index] = calculate_angle(
                point_a[index], point_b[index], point_c[index]
            )
    return values


def _trunk_series(
    shoulder: np.ndarray,
    hip: np.ndarray,
    confidence: np.ndarray,
    min_visibility: float,
) -> np.ndarray:
    values = np.full(len(shoulder), np.nan, dtype=float)
    for index in range(len(values)):
        if confidence[index] >= min_visibility:
            values[index] = calculate_trunk_lean(shoulder[index], hip[index])
    return values


def compute_kinematics(
    landmark_data: pd.DataFrame,
    frame_width: int,
    frame_height: int,
    min_visibility: float = 0.6,
    requested_side: str = "auto",
    smoothing_window: int = 7,
) -> KinematicsResult:
    """Compute 2D knee, hip and trunk signals from long-format pose data."""
    if requested_side.lower() not in {"auto", "left", "right"}:
        raise ValueError("requested_side must be auto, left or right")
    if not 0.0 <= min_visibility <= 1.0:
        raise ValueError("min_visibility must be between 0 and 1")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if landmark_data.empty:
        raise ValueError("landmark_data cannot be empty")

    required_columns = {
        "frame", "timestamp_seconds", "landmark_name", "x", "y", "visibility"
    }
    missing_columns = required_columns.difference(landmark_data.columns)
    if missing_columns:
        raise ValueError(f"Missing landmark columns: {sorted(missing_columns)}")

    pivot = (
        landmark_data.pivot(
            index=["frame", "timestamp_seconds"],
            columns="landmark_name",
            values=["x", "y", "visibility"],
        )
        .sort_index()
    )
    output = pivot.index.to_frame(index=False)
    side_visibility: dict[str, float] = {}
    side_reliable_rate: dict[str, float] = {}

    for side in SIDES:
        shoulder = _point(pivot, f"{side}_shoulder", frame_width, frame_height)
        hip = _point(pivot, f"{side}_hip", frame_width, frame_height)
        knee = _point(pivot, f"{side}_knee", frame_width, frame_height)
        ankle = _point(pivot, f"{side}_ankle", frame_width, frame_height)

        shoulder_visibility = _visibility(pivot, f"{side}_shoulder")
        hip_visibility = _visibility(pivot, f"{side}_hip")
        knee_visibility = _visibility(pivot, f"{side}_knee")
        ankle_visibility = _visibility(pivot, f"{side}_ankle")
        side_confidence = np.min(
            np.vstack(
                (
                    shoulder_visibility,
                    hip_visibility,
                    knee_visibility,
                    ankle_visibility,
                )
            ),
            axis=0,
        )
        knee_confidence = np.min(
            np.vstack((hip_visibility, knee_visibility, ankle_visibility)), axis=0
        )
        hip_confidence = np.min(
            np.vstack((shoulder_visibility, hip_visibility, knee_visibility)), axis=0
        )
        trunk_confidence = np.min(
            np.vstack((shoulder_visibility, hip_visibility)), axis=0
        )

        output[f"{side}_visibility"] = side_confidence
        output[f"{side}_knee_angle_raw_degrees"] = _angle_series(
            hip, knee, ankle, knee_confidence, min_visibility
        )
        output[f"{side}_hip_angle_raw_degrees"] = _angle_series(
            shoulder, hip, knee, hip_confidence, min_visibility
        )
        output[f"{side}_trunk_lean_raw_degrees"] = _trunk_series(
            shoulder, hip, trunk_confidence, min_visibility
        )

        for signal in ("knee_angle", "hip_angle", "trunk_lean"):
            output[f"{side}_{signal}_degrees"] = smooth_signal(
                output[f"{side}_{signal}_raw_degrees"],
                method="savgol",
                window_size=smoothing_window,
            )

        side_visibility[side] = float(np.mean(side_confidence))
        side_reliable_rate[side] = float(np.mean(side_confidence >= min_visibility))

    selected_side = requested_side.lower()
    if selected_side == "auto":
        selected_side = max(SIDES, key=lambda side: side_visibility[side])

    for signal in ("knee_angle", "hip_angle", "trunk_lean"):
        output[f"{signal}_raw_degrees"] = output[
            f"{selected_side}_{signal}_raw_degrees"
        ]
        output[f"{signal}_degrees"] = output[
            f"{selected_side}_{signal}_degrees"
        ]

    output["knee_asymmetry_degrees"] = calculate_asymmetry(
        output["left_knee_angle_degrees"], output["right_knee_angle_degrees"]
    )
    output["hip_asymmetry_degrees"] = calculate_asymmetry(
        output["left_hip_angle_degrees"], output["right_hip_angle_degrees"]
    )
    valid = output[
        ["knee_angle_degrees", "hip_angle_degrees", "trunk_lean_degrees"]
    ].notna().all(axis=1)

    metrics = KinematicsMetrics(
        knee_min_degrees=_finite_stat(output["knee_angle_degrees"], "min"),
        knee_max_degrees=_finite_stat(output["knee_angle_degrees"], "max"),
        knee_rom_degrees=calculate_range_of_motion(output["knee_angle_degrees"]),
        hip_min_degrees=_finite_stat(output["hip_angle_degrees"], "min"),
        hip_max_degrees=_finite_stat(output["hip_angle_degrees"], "max"),
        hip_rom_degrees=calculate_range_of_motion(output["hip_angle_degrees"]),
        trunk_mean_degrees=_finite_stat(output["trunk_lean_degrees"], "mean"),
        trunk_max_degrees=_finite_stat(output["trunk_lean_degrees"], "max"),
        valid_frame_rate=float(valid.mean()),
        knee_asymmetry_mean_degrees=_finite_stat(
            output["knee_asymmetry_degrees"], "mean"
        ),
        knee_asymmetry_max_degrees=_finite_stat(
            output["knee_asymmetry_degrees"], "max"
        ),
    )
    return KinematicsResult(
        data=output,
        selected_side=selected_side,
        side_visibility=side_visibility,
        side_reliable_rate=side_reliable_rate,
        metrics=metrics,
    )

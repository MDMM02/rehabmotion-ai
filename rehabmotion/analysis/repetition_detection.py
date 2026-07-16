from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RepetitionSegment:
    rep_id: int
    start_index: int
    turning_index: int
    end_index: int
    start_time_seconds: float
    turning_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    phase_1_duration_seconds: float
    phase_2_duration_seconds: float
    knee_min_degrees: float
    knee_max_degrees: float
    knee_rom_degrees: float


@dataclass(frozen=True, slots=True)
class RepetitionDetectionResult:
    repetitions: tuple[RepetitionSegment, ...]
    start_threshold_degrees: float
    turning_threshold_degrees: float
    signal_excursion_degrees: float
    exercise_type: str
    warning: str | None = None


def _exercise_key(exercise_type: str) -> str:
    key = exercise_type.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "squat": "squat",
        "sit-to-stand": "sit-to-stand",
        "knee-flexion": "knee-flexion",
    }
    if key not in aliases:
        raise ValueError("exercise_type must be squat, sit-to-stand or knee-flexion")
    return aliases[key]


def detect_repetitions(
    angle_series: Sequence[float],
    timestamps: Sequence[float],
    exercise_type: str = "squat",
    min_signal_excursion_degrees: float = 25.0,
    min_duration_seconds: float = 0.6,
    max_duration_seconds: float = 10.0,
) -> RepetitionDetectionResult:
    """Detect complete knee-angle cycles using adaptive hysteresis thresholds."""
    angles = np.asarray(angle_series, dtype=float)
    time = np.asarray(timestamps, dtype=float)
    if angles.ndim != 1 or time.ndim != 1 or angles.shape != time.shape:
        raise ValueError("angle_series and timestamps must be aligned 1D arrays")
    if angles.size < 5:
        raise ValueError("At least five angle samples are required")
    if np.any(np.diff(time) < 0):
        raise ValueError("timestamps must be monotonically increasing")

    key = _exercise_key(exercise_type)
    finite = angles[np.isfinite(angles)]
    if finite.size < 5:
        return RepetitionDetectionResult(
            repetitions=(),
            start_threshold_degrees=float("nan"),
            turning_threshold_degrees=float("nan"),
            signal_excursion_degrees=0.0,
            exercise_type=key,
            warning="Too few reliable knee-angle samples for repetition detection.",
        )

    # Squat/knee flexion: extension -> flexion -> extension (high-low-high).
    # Sit-to-stand: seated -> standing -> seated (low-high-low), so invert once.
    polarity = -1.0 if key == "sit-to-stand" else 1.0
    working = angles * polarity
    working_finite = working[np.isfinite(working)]
    robust_low = float(np.percentile(working_finite, 5))
    robust_high = float(np.percentile(working_finite, 90))
    excursion = robust_high - robust_low
    if excursion < min_signal_excursion_degrees:
        return RepetitionDetectionResult(
            repetitions=(),
            start_threshold_degrees=robust_high * polarity,
            turning_threshold_degrees=robust_low * polarity,
            signal_excursion_degrees=excursion,
            exercise_type=key,
            warning=(
                "The knee-angle excursion is too small to detect reliable "
                "repetitions."
            ),
        )

    turning_threshold = robust_low + 0.20 * excursion
    start_threshold = robust_low + 0.75 * excursion
    repetitions: list[RepetitionSegment] = []
    state = "waiting"
    armed = False
    last_start_index: int | None = None
    start_index: int | None = None
    turning_index: int | None = None

    for index, value in enumerate(working):
        if not np.isfinite(value):
            state = "waiting"
            armed = False
            last_start_index = None
            start_index = None
            turning_index = None
            continue

        if state == "waiting":
            if value >= start_threshold:
                armed = True
                last_start_index = index
            elif armed and last_start_index is not None:
                start_index = last_start_index
                turning_index = index
                state = "toward_turn"
            continue

        if state == "toward_turn":
            if turning_index is None or value < working[turning_index]:
                turning_index = index
            if value <= turning_threshold:
                state = "returning"
            elif value >= start_threshold:
                # A shallow partial movement returned to its starting posture.
                state = "waiting"
                armed = True
                last_start_index = index
                start_index = None
                turning_index = None
            continue

        if state == "returning":
            if turning_index is None or value < working[turning_index]:
                turning_index = index
            if value < start_threshold:
                continue

            threshold_end_index = index
            assert start_index is not None and turning_index is not None
            expanded_start_index = start_index
            while (
                expanded_start_index > 0
                and np.isfinite(working[expanded_start_index - 1])
                and working[expanded_start_index - 1]
                >= working[expanded_start_index]
            ):
                expanded_start_index -= 1

            end_index = threshold_end_index
            while (
                end_index + 1 < len(working)
                and np.isfinite(working[end_index + 1])
                and working[end_index + 1] >= working[end_index]
            ):
                end_index += 1

            start_index = expanded_start_index
            duration = float(time[end_index] - time[start_index])
            segment_angles = angles[start_index : end_index + 1]
            segment_finite = segment_angles[np.isfinite(segment_angles)]
            knee_min = float(np.min(segment_finite))
            knee_max = float(np.max(segment_finite))
            knee_rom = knee_max - knee_min
            if (
                min_duration_seconds <= duration <= max_duration_seconds
                and knee_rom >= min_signal_excursion_degrees
            ):
                repetitions.append(
                    RepetitionSegment(
                        rep_id=len(repetitions) + 1,
                        start_index=start_index,
                        turning_index=turning_index,
                        end_index=end_index,
                        start_time_seconds=float(time[start_index]),
                        turning_time_seconds=float(time[turning_index]),
                        end_time_seconds=float(time[end_index]),
                        duration_seconds=duration,
                        phase_1_duration_seconds=float(
                            time[turning_index] - time[start_index]
                        ),
                        phase_2_duration_seconds=float(
                            time[end_index] - time[turning_index]
                        ),
                        knee_min_degrees=knee_min,
                        knee_max_degrees=knee_max,
                        knee_rom_degrees=knee_rom,
                    )
                )
            state = "waiting"
            armed = True
            last_start_index = threshold_end_index
            start_index = None
            turning_index = None

    warning = None
    if not repetitions:
        warning = (
            "No complete repetitions were detected. Check the exercise type, "
            "camera view and landmark visibility."
        )
    return RepetitionDetectionResult(
        repetitions=tuple(repetitions),
        start_threshold_degrees=start_threshold * polarity,
        turning_threshold_degrees=turning_threshold * polarity,
        signal_excursion_degrees=excursion,
        exercise_type=key,
        warning=warning,
    )

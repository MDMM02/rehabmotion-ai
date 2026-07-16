import numpy as np

from rehabmotion.analysis.repetition_detection import detect_repetitions


def _squat_signal(repetitions: int = 3) -> tuple[np.ndarray, np.ndarray]:
    partial = [170, 155, 130, 112, 135, 160, 170]
    complete = [170, 160, 140, 110, 80, 110, 140, 160, 170]
    angles = np.asarray(partial + complete * repetitions, dtype=float)
    timestamps = np.arange(len(angles), dtype=float) * 0.2
    return angles, timestamps


def test_squat_detector_ignores_partial_flexion() -> None:
    angles, timestamps = _squat_signal(3)

    result = detect_repetitions(angles, timestamps, exercise_type="squat")

    assert len(result.repetitions) == 3
    assert all(repetition.knee_rom_degrees >= 80 for repetition in result.repetitions)
    assert [repetition.rep_id for repetition in result.repetitions] == [1, 2, 3]


def test_sit_to_stand_detects_low_high_low_cycles() -> None:
    cycle = [90, 105, 130, 160, 175, 160, 130, 105, 90]
    angles = np.asarray(cycle * 2, dtype=float)
    timestamps = np.arange(len(angles), dtype=float) * 0.2

    result = detect_repetitions(
        angles, timestamps, exercise_type="sit-to-stand"
    )

    assert len(result.repetitions) == 2
    assert result.start_threshold_degrees < result.turning_threshold_degrees


def test_flat_signal_returns_warning_instead_of_false_repetitions() -> None:
    angles = np.full(30, 170.0)
    timestamps = np.arange(30, dtype=float) * 0.1

    result = detect_repetitions(angles, timestamps)

    assert not result.repetitions
    assert result.warning is not None

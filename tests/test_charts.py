from app.components.charts import build_kinematics_figure
from rehabmotion.analysis.repetition_detection import detect_repetitions
from rehabmotion.biomechanics.kinematics import compute_kinematics
from tests.test_kinematics import _synthetic_landmarks


def test_kinematics_chart_contains_raw_and_smoothed_signals() -> None:
    result = compute_kinematics(
        _synthetic_landmarks(),
        frame_width=100,
        frame_height=100,
        smoothing_window=5,
    )

    figure = build_kinematics_figure(result.data, result.selected_side)

    assert len(figure.data) == 6


def test_kinematics_chart_marks_detected_repetitions() -> None:
    result = compute_kinematics(
        _synthetic_landmarks(),
        frame_width=100,
        frame_height=100,
        smoothing_window=5,
    )
    detection = detect_repetitions(
        [170, 150, 110, 80, 110, 150, 170, 150, 110, 80, 110, 150, 170],
        [index * 0.2 for index in range(13)],
    )

    figure = build_kinematics_figure(
        result.data, result.selected_side, detection.repetitions
    )

    assert len(figure.layout.shapes) == 6

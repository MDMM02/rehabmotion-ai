import pandas as pd

from rehabmotion.biomechanics.kinematics import compute_kinematics


LANDMARK_POINTS = {
    "left_shoulder": (0.40, 0.20),
    "right_shoulder": (0.60, 0.20),
    "left_hip": (0.40, 0.50),
    "right_hip": (0.60, 0.50),
    "left_knee": (0.40, 0.70),
    "right_knee": (0.60, 0.70),
    "left_ankle": (0.40, 0.90),
    "right_ankle": (0.60, 0.90),
}


def _synthetic_landmarks() -> pd.DataFrame:
    rows = []
    ankle_offsets = (0.00, 0.03, 0.08, 0.12, 0.08, 0.03, 0.00)
    for frame, offset in enumerate(ankle_offsets):
        for name, (x, y) in LANDMARK_POINTS.items():
            if name == "right_ankle":
                x += offset
            rows.append(
                {
                    "frame": frame,
                    "timestamp_seconds": frame / 10,
                    "landmark_name": name,
                    "x": x,
                    "y": y,
                    "visibility": 0.95 if name.startswith("right") else 0.30,
                }
            )
    return pd.DataFrame(rows)


def test_auto_side_uses_more_visible_body_side() -> None:
    result = compute_kinematics(
        _synthetic_landmarks(),
        frame_width=100,
        frame_height=100,
        requested_side="auto",
        smoothing_window=5,
    )

    assert result.selected_side == "right"
    assert result.side_reliable_rate["right"] == 1.0
    assert result.side_reliable_rate["left"] == 0.0
    assert result.metrics.valid_frame_rate == 1.0
    assert result.metrics.knee_rom_degrees > 0
    assert result.data["left_knee_angle_degrees"].isna().all()


def test_explicit_low_visibility_side_keeps_angles_missing() -> None:
    result = compute_kinematics(
        _synthetic_landmarks(),
        frame_width=100,
        frame_height=100,
        requested_side="left",
        smoothing_window=5,
    )

    assert result.selected_side == "left"
    assert result.metrics.valid_frame_rate == 0.0

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import pandas as pd


POSE_LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

# Connections from the official MediaPipe Pose Landmarker topology.
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18),
    (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
    (27, 31), (28, 32),
)

REQUIRED_REHAB_LANDMARKS = (11, 12, 23, 24, 25, 26, 27, 28)


@dataclass(frozen=True, slots=True)
class LandmarkRecord:
    frame: int
    timestamp_seconds: float
    landmark_id: int
    landmark_name: str
    x: float
    y: float
    z: float
    visibility: float
    presence: float


def _score(value: float | None) -> float:
    return float(value) if value is not None else 0.0


def records_from_landmarks(
    landmarks: Sequence[Any], frame: int, timestamp_seconds: float
) -> list[LandmarkRecord]:
    """Convert MediaPipe landmarks to serializable frame-level records."""
    records: list[LandmarkRecord] = []
    for landmark_id, landmark in enumerate(landmarks):
        if landmark_id >= len(POSE_LANDMARK_NAMES):
            break
        records.append(
            LandmarkRecord(
                frame=frame,
                timestamp_seconds=timestamp_seconds,
                landmark_id=landmark_id,
                landmark_name=POSE_LANDMARK_NAMES[landmark_id],
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                visibility=_score(getattr(landmark, "visibility", None)),
                presence=_score(getattr(landmark, "presence", None)),
            )
        )
    return records


def records_to_dataframe(records: Sequence[LandmarkRecord]) -> pd.DataFrame:
    """Build a stable long-format DataFrame for preview and CSV export."""
    columns = [field.name for field in LandmarkRecord.__dataclass_fields__.values()]
    return pd.DataFrame((asdict(record) for record in records), columns=columns)

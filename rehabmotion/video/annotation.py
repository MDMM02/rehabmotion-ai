from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from rehabmotion.pose.landmarks import POSE_CONNECTIONS


def _pixel_coordinates(landmark: Any, width: int, height: int) -> tuple[int, int]:
    x = int(round(float(landmark.x) * width))
    y = int(round(float(landmark.y) * height))
    return x, y


def _visibility(landmark: Any) -> float:
    value = getattr(landmark, "visibility", None)
    return float(value) if value is not None else 0.0


def annotate_pose_frame(
    frame_bgr: np.ndarray,
    landmarks: Sequence[Any],
    min_visibility: float = 0.6,
    label: str | None = None,
) -> np.ndarray:
    """Draw the MediaPipe pose skeleton and return an RGB preview image."""
    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]

    for start, end in POSE_CONNECTIONS:
        if start >= len(landmarks) or end >= len(landmarks):
            continue
        start_landmark = landmarks[start]
        end_landmark = landmarks[end]
        if min(_visibility(start_landmark), _visibility(end_landmark)) < 0.1:
            continue
        cv2.line(
            annotated,
            _pixel_coordinates(start_landmark, width, height),
            _pixel_coordinates(end_landmark, width, height),
            (30, 210, 255),
            2,
            cv2.LINE_AA,
        )

    for landmark in landmarks:
        visibility = _visibility(landmark)
        if visibility < 0.1:
            continue
        color = (60, 220, 80) if visibility >= min_visibility else (0, 140, 255)
        cv2.circle(
            annotated,
            _pixel_coordinates(landmark, width, height),
            4,
            color,
            -1,
            cv2.LINE_AA,
        )

    if label:
        cv2.rectangle(annotated, (8, 8), (min(width - 8, 430), 42), (18, 18, 18), -1)
        cv2.putText(
            annotated,
            label,
            (16, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

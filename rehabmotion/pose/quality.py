from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rehabmotion.pose.landmarks import REQUIRED_REHAB_LANDMARKS


@dataclass(frozen=True, slots=True)
class LandmarkQuality:
    detected: bool
    reliable: bool
    mean_visibility: float
    minimum_visibility: float
    low_confidence_landmarks: tuple[int, ...]


def _visibility(landmark: Any) -> float:
    visibility = getattr(landmark, "visibility", None)
    if visibility is not None:
        return float(visibility)
    presence = getattr(landmark, "presence", None)
    return float(presence) if presence is not None else 0.0


def assess_landmark_quality(
    landmarks: Sequence[Any] | None,
    min_visibility: float = 0.6,
    required_indices: Sequence[int] = REQUIRED_REHAB_LANDMARKS,
) -> LandmarkQuality:
    """Assess visibility of shoulders, hips, knees and ankles."""
    if not landmarks or len(landmarks) <= max(required_indices):
        return LandmarkQuality(False, False, 0.0, 0.0, tuple(required_indices))

    visibility_scores = [_visibility(landmarks[index]) for index in required_indices]
    low_confidence = tuple(
        index
        for index, visibility in zip(required_indices, visibility_scores)
        if visibility < min_visibility
    )
    return LandmarkQuality(
        detected=True,
        reliable=not low_confidence,
        mean_visibility=sum(visibility_scores) / len(visibility_scores),
        minimum_visibility=min(visibility_scores),
        low_confidence_landmarks=low_confidence,
    )

from types import SimpleNamespace

from rehabmotion.pose.quality import assess_landmark_quality


def _landmarks(visibility: float) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=visibility, presence=1.0)
        for _ in range(33)
    ]


def test_required_landmarks_are_reliable_above_threshold() -> None:
    quality = assess_landmark_quality(_landmarks(0.8), min_visibility=0.6)

    assert quality.detected
    assert quality.reliable
    assert quality.mean_visibility == 0.8
    assert not quality.low_confidence_landmarks


def test_one_hidden_required_landmark_marks_frame_low_confidence() -> None:
    landmarks = _landmarks(0.9)
    landmarks[25].visibility = 0.2

    quality = assess_landmark_quality(landmarks, min_visibility=0.6)

    assert quality.detected
    assert not quality.reliable
    assert quality.low_confidence_landmarks == (25,)


def test_missing_pose_is_not_reliable() -> None:
    quality = assess_landmark_quality(None)

    assert not quality.detected
    assert not quality.reliable

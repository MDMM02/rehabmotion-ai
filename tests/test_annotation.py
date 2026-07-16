from types import SimpleNamespace

import numpy as np

from rehabmotion.video.annotation import annotate_pose_frame


def test_annotation_returns_rgb_image_with_drawing() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    landmarks = [
        SimpleNamespace(x=0.25 + index / 100, y=0.5, visibility=0.9)
        for index in range(33)
    ]

    annotated = annotate_pose_frame(frame, landmarks, label="pose")

    assert annotated.shape == frame.shape
    assert np.any(annotated != 0)

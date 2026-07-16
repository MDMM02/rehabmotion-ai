from pathlib import Path

import cv2
import numpy as np
import pytest

from rehabmotion.pose.estimator import analyze_video_pose
from rehabmotion.utils.config import POSE_MODEL_PATH


def test_real_model_processes_video_without_pose(tmp_path: Path) -> None:
    """Smoke-test the installed MediaPipe runtime with a tiny blank video."""
    if not POSE_MODEL_PATH.is_file():
        pytest.skip("MediaPipe task model is not cached locally")

    video_path = tmp_path / "blank.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (160, 120),
    )
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    result = analyze_video_pose(
        video_path,
        POSE_MODEL_PATH,
        min_visibility=0.6,
        target_fps=10.0,
    )

    assert result.processed_frame_count == 5
    assert result.detected_frame_count == 0
    assert result.landmarks_dataframe().empty

from pathlib import Path

import numpy as np
import pytest

from rehabmotion.video.reader import read_video_metadata
from rehabmotion.video.writer import write_bgr_video


def test_write_bgr_video_creates_readable_clip(tmp_path: Path) -> None:
    frames = [
        np.full((48, 64, 3), fill_value=value, dtype=np.uint8)
        for value in (0, 80, 160)
    ]

    path = write_bgr_video(
        frames,
        tmp_path / "preview.avi",
        fps=5.0,
        codec="MJPG",
    )
    metadata = read_video_metadata(path)

    assert metadata.frame_count == 3
    assert metadata.fps == pytest.approx(5.0)
    assert (metadata.width, metadata.height) == (64, 48)


def test_write_bgr_video_rejects_mismatched_frames(tmp_path: Path) -> None:
    frames = [
        np.zeros((48, 64, 3), dtype=np.uint8),
        np.zeros((40, 64, 3), dtype=np.uint8),
    ]

    with pytest.raises(ValueError, match="same dimensions"):
        write_bgr_video(frames, tmp_path / "bad.avi", fps=5.0, codec="MJPG")

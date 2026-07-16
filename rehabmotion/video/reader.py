from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from rehabmotion.utils.exceptions import InvalidVideoError


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Basic properties read from a video container."""

    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Read and validate basic metadata from a local video file."""
    path = Path(video_path)
    if not path.is_file():
        raise InvalidVideoError("The selected video file does not exist.")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise InvalidVideoError(
                "This video cannot be read. Please upload a valid MP4, MOV or AVI file."
            )

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise InvalidVideoError(
                "The file opened, but its video metadata is incomplete or invalid."
            )

        return VideoMetadata(
            duration_seconds=frame_count / fps,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )
    finally:
        capture.release()


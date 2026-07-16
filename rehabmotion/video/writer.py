from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np


def write_bgr_video(
    frames: Sequence[np.ndarray],
    output_path: str | Path,
    fps: float,
    codec: str = "mp4v",
) -> Path:
    """Write equally sized BGR frames to a local video file."""
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if not frames:
        raise ValueError("at least one frame is required")
    if len(codec) != 4:
        raise ValueError("codec must contain exactly four characters")

    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("frames must be BGR images with three channels")
    height, width = first.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise OSError(f"Could not create video file: {path}")

    try:
        for frame in frames:
            image = np.asarray(frame)
            if image.shape != first.shape:
                raise ValueError("all frames must have the same dimensions")
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            writer.write(image)
    finally:
        writer.release()
    return path

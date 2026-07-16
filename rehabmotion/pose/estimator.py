from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.error import URLError
from urllib.request import urlretrieve

import cv2
import numpy as np
import pandas as pd

from rehabmotion.pose.landmarks import (
    LandmarkRecord,
    records_from_landmarks,
    records_to_dataframe,
)
from rehabmotion.pose.quality import assess_landmark_quality
from rehabmotion.utils.config import POSE_MODEL_PATH, POSE_MODEL_URL
from rehabmotion.utils.exceptions import PoseAnalysisError, PoseModelError
from rehabmotion.video.annotation import annotate_pose_frame


@dataclass(frozen=True, slots=True)
class FramePoseQuality:
    frame: int
    timestamp_seconds: float
    pose_detected: bool
    reliable: bool
    mean_visibility: float
    minimum_visibility: float


@dataclass(slots=True)
class PoseAnalysisResult:
    landmarks: list[LandmarkRecord]
    frame_quality: list[FramePoseQuality]
    preview_rgb: np.ndarray | None
    source_frame_count: int
    processed_frame_count: int
    detected_frame_count: int
    reliable_frame_count: int
    effective_fps: float
    frame_width: int
    frame_height: int

    @property
    def detection_rate(self) -> float:
        if not self.processed_frame_count:
            return 0.0
        return self.detected_frame_count / self.processed_frame_count

    @property
    def reliable_rate(self) -> float:
        if not self.detected_frame_count:
            return 0.0
        return self.reliable_frame_count / self.detected_frame_count

    @property
    def mean_visibility(self) -> float:
        detected = [item.mean_visibility for item in self.frame_quality if item.pose_detected]
        return sum(detected) / len(detected) if detected else 0.0

    def landmarks_dataframe(self) -> pd.DataFrame:
        return records_to_dataframe(self.landmarks)

    def quality_dataframe(self) -> pd.DataFrame:
        columns = [field.name for field in FramePoseQuality.__dataclass_fields__.values()]
        return pd.DataFrame(
            (asdict(item) for item in self.frame_quality), columns=columns
        )


def ensure_pose_model(
    model_path: str | Path = POSE_MODEL_PATH,
    model_url: str = POSE_MODEL_URL,
) -> Path:
    """Download the official MediaPipe lite model once and return its path."""
    path = Path(model_path)
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".download")
    try:
        urlretrieve(model_url, temporary_path)
        if temporary_path.stat().st_size <= 1_000_000:
            raise PoseModelError("The downloaded pose model is incomplete.")
        temporary_path.replace(path)
    except PoseModelError:
        temporary_path.unlink(missing_ok=True)
        raise
    except (OSError, URLError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise PoseModelError(
            "The MediaPipe pose model could not be downloaded. "
            "Check the internet connection and try again."
        ) from exc
    return path


def _mediapipe_modules() -> tuple[Any, Any, Any]:
    # MediaPipe imports Matplotlib; use a writable cache in restricted runtimes.
    matplotlib_cache = Path(tempfile.gettempdir()) / "rehabmotion-matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise PoseAnalysisError(
            "MediaPipe is not installed. Install the project requirements and try again."
        ) from exc
    return mp, python, vision


def analyze_video_pose(
    video_path: str | Path,
    model_path: str | Path,
    min_visibility: float = 0.6,
    target_fps: float = 10.0,
) -> PoseAnalysisResult:
    """Run single-person pose estimation on sampled frames from a video."""
    if not 0.0 <= min_visibility <= 1.0:
        raise ValueError("min_visibility must be between 0 and 1")
    if target_fps <= 0:
        raise ValueError("target_fps must be greater than zero")

    path = Path(video_path)
    model = Path(model_path)
    if not path.is_file():
        raise PoseAnalysisError("The selected video file does not exist.")
    if not model.is_file():
        raise PoseModelError("The MediaPipe pose model is missing.")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PoseAnalysisError("The video could not be opened for pose analysis.")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (
        source_fps <= 0
        or source_frame_count <= 0
        or frame_width <= 0
        or frame_height <= 0
    ):
        capture.release()
        raise PoseAnalysisError("The video FPS or frame count is invalid.")

    frame_step = max(1, int(round(source_fps / min(target_fps, source_fps))))
    effective_fps = source_fps / frame_step
    mp, python, vision = _mediapipe_modules()
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    landmark_records: list[LandmarkRecord] = []
    frame_quality: list[FramePoseQuality] = []
    preview_rgb: np.ndarray | None = None
    best_preview_score = -1.0
    processed_frames = 0
    detected_frames = 0
    reliable_frames = 0
    frame_index = 0
    last_timestamp_ms = -1

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame_bgr = capture.read()
                if not success:
                    break
                if frame_index % frame_step:
                    frame_index += 1
                    continue

                timestamp_ms = int(round(frame_index * 1000 / source_fps))
                timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect_for_video(media_image, timestamp_ms)
                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                quality = assess_landmark_quality(landmarks, min_visibility)

                processed_frames += 1
                detected_frames += int(quality.detected)
                reliable_frames += int(quality.reliable)
                timestamp_seconds = timestamp_ms / 1000.0
                frame_quality.append(
                    FramePoseQuality(
                        frame=frame_index,
                        timestamp_seconds=timestamp_seconds,
                        pose_detected=quality.detected,
                        reliable=quality.reliable,
                        mean_visibility=quality.mean_visibility,
                        minimum_visibility=quality.minimum_visibility,
                    )
                )

                if landmarks:
                    landmark_records.extend(
                        records_from_landmarks(
                            landmarks, frame_index, timestamp_seconds
                        )
                    )
                    if quality.mean_visibility > best_preview_score:
                        status = "reliable" if quality.reliable else "low confidence"
                        preview_rgb = annotate_pose_frame(
                            frame_bgr,
                            landmarks,
                            min_visibility=min_visibility,
                            label=(
                                f"t={timestamp_seconds:.2f}s | "
                                f"visibility={quality.mean_visibility:.2f} | {status}"
                            ),
                        )
                        best_preview_score = quality.mean_visibility

                frame_index += 1
    except (RuntimeError, ValueError) as exc:
        raise PoseAnalysisError(
            "MediaPipe could not complete pose analysis on this video."
        ) from exc
    finally:
        capture.release()

    return PoseAnalysisResult(
        landmarks=landmark_records,
        frame_quality=frame_quality,
        preview_rgb=preview_rgb,
        source_frame_count=source_frame_count,
        processed_frame_count=processed_frames,
        detected_frame_count=detected_frames,
        reliable_frame_count=reliable_frames,
        effective_fps=effective_fps,
        frame_width=frame_width,
        frame_height=frame_height,
    )

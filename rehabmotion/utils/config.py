"""Project-wide configuration values."""

from pathlib import Path

SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi"})

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSE_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

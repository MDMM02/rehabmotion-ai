class RehabMotionError(Exception):
    """Base exception for expected RehabMotion errors."""


class InvalidVideoError(RehabMotionError):
    """Raised when a video is absent, unreadable or has invalid metadata."""


class PoseModelError(RehabMotionError):
    """Raised when the pose model cannot be found or downloaded."""


class PoseAnalysisError(RehabMotionError):
    """Raised when pose analysis cannot be completed."""

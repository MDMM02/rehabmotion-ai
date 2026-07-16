from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rehabmotion.analysis.movement_metrics import RepetitionSummary
from rehabmotion.analysis.repetition_detection import RepetitionDetectionResult
from rehabmotion.biomechanics.kinematics import KinematicsResult
from rehabmotion.pose.estimator import PoseAnalysisResult
from rehabmotion.video.reader import VideoMetadata


@dataclass(frozen=True, slots=True)
class VideoReportInfo:
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class QualityReportMetrics:
    pose_detection_rate: float
    bilateral_reliable_rate: float
    mean_visibility: float
    usable_kinematics_rate: float


@dataclass(frozen=True, slots=True)
class MovementReportMetrics:
    knee_min_degrees: float
    knee_max_degrees: float
    knee_rom_degrees: float
    hip_min_degrees: float
    hip_max_degrees: float
    hip_rom_degrees: float
    trunk_mean_degrees: float
    trunk_max_degrees: float
    knee_asymmetry_mean_degrees: float
    knee_asymmetry_max_degrees: float
    detected_repetitions: int
    mean_repetition_duration_seconds: float
    duration_cv: float
    tempo_regularity: str
    mean_repetition_knee_rom_degrees: float


@dataclass(frozen=True, slots=True)
class MovementReportData:
    source_name: str
    exercise_type: str
    analyzed_side: str
    video: VideoReportInfo
    quality: QualityReportMetrics
    movement: MovementReportMetrics
    kinematics: pd.DataFrame
    repetitions: pd.DataFrame
    start_threshold_degrees: float
    turning_threshold_degrees: float
    signal_excursion_degrees: float
    detection_warning: str | None


def build_report_data(
    *,
    source_name: str,
    exercise_type: str,
    metadata: VideoMetadata,
    pose: PoseAnalysisResult,
    kinematics: KinematicsResult,
    detection: RepetitionDetectionResult,
    repetition_metrics: pd.DataFrame,
    repetition_summary: RepetitionSummary,
) -> MovementReportData:
    """Assemble an immutable snapshot used by all report exporters."""
    metrics = kinematics.metrics
    chart_columns = [
        "timestamp_seconds",
        "knee_angle_degrees",
        "hip_angle_degrees",
        "trunk_lean_degrees",
    ]
    return MovementReportData(
        source_name=source_name,
        exercise_type=exercise_type,
        analyzed_side=kinematics.selected_side,
        video=VideoReportInfo(
            duration_seconds=metadata.duration_seconds,
            fps=metadata.fps,
            frame_count=metadata.frame_count,
            width=metadata.width,
            height=metadata.height,
        ),
        quality=QualityReportMetrics(
            pose_detection_rate=pose.detection_rate,
            bilateral_reliable_rate=pose.reliable_rate,
            mean_visibility=pose.mean_visibility,
            usable_kinematics_rate=metrics.valid_frame_rate,
        ),
        movement=MovementReportMetrics(
            knee_min_degrees=metrics.knee_min_degrees,
            knee_max_degrees=metrics.knee_max_degrees,
            knee_rom_degrees=metrics.knee_rom_degrees,
            hip_min_degrees=metrics.hip_min_degrees,
            hip_max_degrees=metrics.hip_max_degrees,
            hip_rom_degrees=metrics.hip_rom_degrees,
            trunk_mean_degrees=metrics.trunk_mean_degrees,
            trunk_max_degrees=metrics.trunk_max_degrees,
            knee_asymmetry_mean_degrees=metrics.knee_asymmetry_mean_degrees,
            knee_asymmetry_max_degrees=metrics.knee_asymmetry_max_degrees,
            detected_repetitions=repetition_summary.detected_repetitions,
            mean_repetition_duration_seconds=(
                repetition_summary.mean_duration_seconds
            ),
            duration_cv=repetition_summary.duration_cv,
            tempo_regularity=repetition_summary.tempo_regularity,
            mean_repetition_knee_rom_degrees=(
                repetition_summary.mean_knee_rom_degrees
            ),
        ),
        kinematics=kinematics.data.loc[:, chart_columns].copy(),
        repetitions=repetition_metrics.copy(),
        start_threshold_degrees=detection.start_threshold_degrees,
        turning_threshold_degrees=detection.turning_threshold_degrees,
        signal_excursion_degrees=detection.signal_excursion_degrees,
        detection_warning=detection.warning,
    )

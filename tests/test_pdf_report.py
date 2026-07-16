from datetime import datetime, timezone

import numpy as np
import pandas as pd

from rehabmotion.analysis.report_data import (
    MovementReportData,
    MovementReportMetrics,
    QualityReportMetrics,
    VideoReportInfo,
)
from rehabmotion.export.pdf_report import DISCLAIMER, generate_pdf_report


def _sample_report() -> MovementReportData:
    timestamps = np.linspace(0.0, 6.0, 61)
    cycle = np.sin(timestamps * np.pi / 1.5)
    kinematics = pd.DataFrame(
        {
            "timestamp_seconds": timestamps,
            "knee_angle_degrees": 130.0 + 42.0 * cycle,
            "hip_angle_degrees": 115.0 + 32.0 * cycle,
            "trunk_lean_degrees": 15.0 - 8.0 * cycle,
        }
    )
    repetitions = pd.DataFrame(
        [
            {
                "rep": 1,
                "start_time_seconds": 0.0,
                "turning_time_seconds": 1.5,
                "end_time_seconds": 3.0,
                "duration_seconds": 3.0,
                "descent_duration_seconds": 1.5,
                "ascent_duration_seconds": 1.5,
                "knee_rom_degrees": 84.0,
                "hip_rom_degrees": 64.0,
                "max_trunk_lean_degrees": 23.0,
                "mean_knee_asymmetry_degrees": 3.1,
            },
            {
                "rep": 2,
                "start_time_seconds": 3.0,
                "turning_time_seconds": 4.5,
                "end_time_seconds": 6.0,
                "duration_seconds": 3.0,
                "descent_duration_seconds": 1.5,
                "ascent_duration_seconds": 1.5,
                "knee_rom_degrees": 82.0,
                "hip_rom_degrees": 62.0,
                "max_trunk_lean_degrees": 22.0,
                "mean_knee_asymmetry_degrees": 3.4,
            },
        ]
    )
    return MovementReportData(
        source_name="sample_squat.mp4",
        exercise_type="Squat",
        analyzed_side="right",
        video=VideoReportInfo(6.0, 30.0, 180, 1280, 720),
        quality=QualityReportMetrics(0.98, 0.84, 0.91, 0.95),
        movement=MovementReportMetrics(
            knee_min_degrees=88.0,
            knee_max_degrees=172.0,
            knee_rom_degrees=84.0,
            hip_min_degrees=74.0,
            hip_max_degrees=138.0,
            hip_rom_degrees=64.0,
            trunk_mean_degrees=15.4,
            trunk_max_degrees=23.0,
            knee_asymmetry_mean_degrees=3.2,
            knee_asymmetry_max_degrees=7.1,
            detected_repetitions=2,
            mean_repetition_duration_seconds=3.0,
            duration_cv=0.0,
            tempo_regularity="high",
            mean_repetition_knee_rom_degrees=83.0,
        ),
        kinematics=kinematics,
        repetitions=repetitions,
        start_threshold_degrees=151.0,
        turning_threshold_degrees=102.0,
        signal_excursion_degrees=84.0,
        detection_warning=None,
    )


def test_pdf_report_is_a_complete_multi_page_document() -> None:
    report = generate_pdf_report(
        _sample_report(),
        generated_at=datetime(2026, 7, 16, 12, 30, tzinfo=timezone.utc),
    )

    assert report.startswith(b"%PDF-")
    assert report.rstrip().endswith(b"%%EOF")
    assert len(report) > 50_000
    assert DISCLAIMER.startswith("This prototype is for educational")

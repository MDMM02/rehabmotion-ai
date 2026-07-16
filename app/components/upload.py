from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.charts import build_kinematics_figure
from rehabmotion.analysis.movement_metrics import (
    RepetitionSummary,
    compute_repetition_metrics,
)
from rehabmotion.analysis.report_data import MovementReportData, build_report_data
from rehabmotion.analysis.repetition_detection import (
    RepetitionDetectionResult,
    detect_repetitions,
)
from rehabmotion.biomechanics.kinematics import KinematicsResult, compute_kinematics
from rehabmotion.export.csv_exporter import dataframe_to_csv_bytes
from rehabmotion.export.pdf_report import generate_pdf_report
from rehabmotion.pose.estimator import (
    PoseAnalysisResult,
    analyze_video_pose,
    ensure_pose_model,
)
from rehabmotion.utils.exceptions import (
    InvalidVideoError,
    PoseAnalysisError,
    PoseModelError,
)
from rehabmotion.video.reader import VideoMetadata, read_video_metadata


ALLOWED_VIDEO_TYPES = ("mp4", "mov", "avi")
SESSION_RESULT_KEY = "rehabmotion_pose_result"
SESSION_SIGNATURE_KEY = "rehabmotion_pose_signature"


def _format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes:
        return f"{int(minutes)} min {remaining_seconds:.1f} s"
    return f"{remaining_seconds:.1f} s"


def _format_degrees(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.1f}°"


def _format_seconds(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2f} s"


def _exercise_key(exercise_type: str) -> str:
    return exercise_type.lower().replace(" ", "-")


def _analysis_signature(
    video_bytes: bytes, min_visibility: float, target_fps: float
) -> str:
    digest = hashlib.sha256(video_bytes).hexdigest()
    return f"{digest}:{min_visibility:.2f}:{target_fps:.1f}"


def _render_metadata(metadata: VideoMetadata) -> None:
    st.subheader("Video information")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", _format_duration(metadata.duration_seconds))
    col2.metric("FPS", f"{metadata.fps:.2f}")
    col3.metric("Frames", f"{metadata.frame_count:,}")
    col4.metric("Resolution", f"{metadata.width} × {metadata.height}")


@st.cache_data(show_spinner=False)
def _analyze_uploaded_video(
    video_bytes: bytes,
    suffix: str,
    min_visibility: float,
    target_fps: float,
) -> PoseAnalysisResult:
    """Cache MediaPipe inference independently from dashboard interactions."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_video:
            temp_video.write(video_bytes)
            temp_path = Path(temp_video.name)

        return analyze_video_pose(
            temp_path,
            ensure_pose_model(),
            min_visibility=min_visibility,
            target_fps=target_fps,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _generate_cached_pdf(report: MovementReportData) -> bytes:
    """Cache the rendered report while dashboard inputs remain unchanged."""
    return generate_pdf_report(report)


def _render_summary(
    pose: PoseAnalysisResult,
    kinematics: KinematicsResult,
    repetitions: RepetitionSummary,
    exercise_type: str,
    metadata: VideoMetadata,
) -> None:
    st.subheader("Analysis summary")
    first_row = st.columns(5)
    first_row[0].metric("Exercise", exercise_type)
    first_row[1].metric("Duration", _format_duration(metadata.duration_seconds))
    first_row[2].metric("Repetitions", repetitions.detected_repetitions)
    first_row[3].metric("Analyzed side", kinematics.selected_side.title())
    first_row[4].metric("Pose detected", f"{pose.detection_rate:.0%}")

    second_row = st.columns(5)
    second_row[0].metric(
        "Knee ROM", _format_degrees(kinematics.metrics.knee_rom_degrees)
    )
    second_row[1].metric(
        "Hip ROM", _format_degrees(kinematics.metrics.hip_rom_degrees)
    )
    second_row[2].metric(
        "Max trunk lean",
        _format_degrees(kinematics.metrics.trunk_max_degrees),
    )
    second_row[3].metric(
        "Mean rep duration", _format_seconds(repetitions.mean_duration_seconds)
    )
    second_row[4].metric("Tempo", repetitions.tempo_regularity.title())


def _render_overview(
    pose: PoseAnalysisResult,
    kinematics: KinematicsResult,
    requested_side: str,
) -> None:
    preview_column, quality_column = st.columns((3, 2))
    with preview_column:
        st.subheader("Pose preview")
        if pose.preview_rgb is not None:
            st.image(
                pose.preview_rgb,
                channels="RGB",
                caption="Green: reliable landmark | Orange: low confidence",
                use_container_width=True,
            )

    with quality_column:
        st.subheader("Data quality")
        left, right = st.columns(2)
        left.metric("Detected frames", f"{pose.detection_rate:.0%}")
        right.metric("Bilateral reliability", f"{pose.reliable_rate:.0%}")
        left.metric("Mean visibility", f"{pose.mean_visibility:.2f}")
        right.metric("Usable kinematics", f"{kinematics.metrics.valid_frame_rate:.0%}")
        st.caption(
            f"{pose.detected_frame_count}/{pose.processed_frame_count} sampled "
            "frames contain a detected pose."
        )

        if requested_side.lower() == "auto":
            st.info(
                f"Auto selected **{kinematics.selected_side}**: visibility "
                f"left {kinematics.side_visibility['left']:.2f}, "
                f"right {kinematics.side_visibility['right']:.2f}."
            )
        if pose.reliable_rate < 0.5:
            st.warning(
                "Bilateral reliability is low. In side view this often means the "
                "far leg is hidden; side-specific kinematics can still be usable."
            )

    st.info(
        "This overview summarizes an educational 2D estimate. Review the curves "
        "and visibility before interpreting any movement metric."
    )


def _render_kinematics(
    kinematics: KinematicsResult,
    detection: RepetitionDetectionResult,
) -> None:
    st.subheader("2D kinematics")
    metrics = kinematics.metrics
    cards = st.columns(4)
    cards[0].metric("Knee ROM", _format_degrees(metrics.knee_rom_degrees))
    cards[1].metric("Hip ROM", _format_degrees(metrics.hip_rom_degrees))
    cards[2].metric("Max trunk lean", _format_degrees(metrics.trunk_max_degrees))
    cards[3].metric("Usable frames", f"{metrics.valid_frame_rate:.0%}")

    st.plotly_chart(
        build_kinematics_figure(
            kinematics.data,
            kinematics.selected_side,
            repetitions=detection.repetitions,
        ),
        use_container_width=True,
    )
    st.caption(
        "Gray: raw values. Green: Savitzky-Golay smoothing. Shaded areas and "
        "labels indicate detected repetitions; low-confidence gaps remain empty."
    )

    if metrics.valid_frame_rate < 0.5:
        st.warning(
            "Fewer than half of the frames have reliable kinematics. Try Auto "
            "or select the side facing the camera."
        )
    if pd.isna(metrics.knee_asymmetry_mean_degrees):
        st.info(
            "Asymmetry is unavailable because both legs are not simultaneously "
            "reliable. This is common in strict side-view videos."
        )
    else:
        st.metric(
            "Mean knee asymmetry",
            _format_degrees(metrics.knee_asymmetry_mean_degrees),
        )


def _render_repetitions(
    detection: RepetitionDetectionResult,
    metrics: pd.DataFrame,
    summary: RepetitionSummary,
) -> None:
    st.subheader("Repetition analysis")
    st.caption(
        "Complete-cycle heuristic: partial movements that do not cross both "
        "adaptive knee-angle thresholds are not counted."
    )
    if not detection.repetitions:
        st.warning(detection.warning or "No complete repetitions were detected.")
        return

    cards = st.columns(4)
    cards[0].metric("Detected repetitions", summary.detected_repetitions)
    cards[1].metric("Mean duration", _format_seconds(summary.mean_duration_seconds))
    cards[2].metric(
        "Mean knee ROM", _format_degrees(summary.mean_knee_rom_degrees)
    )
    cards[3].metric(
        "Tempo regularity",
        summary.tempo_regularity.title(),
        help=f"Duration coefficient of variation: {summary.duration_cv:.1%}",
    )
    st.dataframe(metrics.round(2), use_container_width=True, hide_index=True)
    st.caption(
        f"Adaptive thresholds: start/return "
        f"{detection.start_threshold_degrees:.1f}°, turning point "
        f"{detection.turning_threshold_degrees:.1f}°."
    )


def _render_data_exports(
    pose: PoseAnalysisResult,
    kinematics: KinematicsResult,
    detection: RepetitionDetectionResult,
    repetition_metrics: pd.DataFrame,
    repetition_summary: RepetitionSummary,
    metadata: VideoMetadata,
    exercise_type: str,
    original_name: str,
) -> None:
    st.subheader("Data and exports")
    stem = Path(original_name).stem
    landmarks = pose.landmarks_dataframe()
    quality = pose.quality_dataframe()

    report = build_report_data(
        source_name=original_name,
        exercise_type=exercise_type,
        metadata=metadata,
        pose=pose,
        kinematics=kinematics,
        detection=detection,
        repetition_metrics=repetition_metrics,
        repetition_summary=repetition_summary,
    )
    with st.spinner("Preparing the PDF report..."):
        pdf_bytes = _generate_cached_pdf(report)

    export_columns = st.columns(4)
    export_columns[0].download_button(
        "Landmarks CSV",
        data=dataframe_to_csv_bytes(landmarks),
        file_name=f"{stem}_pose_landmarks.csv",
        mime="text/csv",
        on_click="ignore",
        use_container_width=True,
    )
    export_columns[1].download_button(
        "Kinematics CSV",
        data=dataframe_to_csv_bytes(kinematics.data),
        file_name=f"{stem}_kinematics.csv",
        mime="text/csv",
        on_click="ignore",
        use_container_width=True,
    )
    export_columns[2].download_button(
        "Repetitions CSV",
        data=dataframe_to_csv_bytes(repetition_metrics),
        file_name=f"{stem}_repetitions.csv",
        mime="text/csv",
        on_click="ignore",
        use_container_width=True,
        disabled=repetition_metrics.empty,
    )
    export_columns[3].download_button(
        "PDF report",
        data=pdf_bytes,
        file_name=f"{stem}_movement_report.pdf",
        mime="application/pdf",
        on_click="ignore",
        use_container_width=True,
    )
    st.caption(
        "The PDF includes video information, quality indicators, global and "
        "per-repetition metrics, joint-angle curves and the medical disclaimer."
    )

    landmark_tab, kinematics_tab, repetition_tab, quality_tab = st.tabs(
        ["Landmarks", "Kinematics", "Repetitions", "Frame quality"]
    )
    with landmark_tab:
        st.dataframe(landmarks.head(500), use_container_width=True, hide_index=True)
        st.caption(f"Previewing 500 of {len(landmarks):,} landmark rows.")
    with kinematics_tab:
        st.dataframe(kinematics.data, use_container_width=True, hide_index=True)
    with repetition_tab:
        if repetition_metrics.empty:
            st.info("No complete repetition data is available for export.")
        else:
            st.dataframe(
                repetition_metrics.round(3),
                use_container_width=True,
                hide_index=True,
            )
    with quality_tab:
        st.dataframe(quality, use_container_width=True, hide_index=True)


def _render_limitations() -> None:
    st.subheader("Limitations and safe use")
    st.warning(
        "This prototype is for educational and R&D purposes only. It is not a "
        "certified medical device and must not guide diagnosis or treatment."
    )
    st.markdown(
        """
- Estimates are derived from a single 2D camera and depend on camera placement.
- Out-of-plane movement, occlusion, loose clothing and poor lighting reduce accuracy.
- ROM values are not equivalent to clinical goniometry or 3D motion capture.
- Repetition thresholds and tempo labels are heuristics and require visual review.
- Left-right asymmetry is unavailable when both sides are not simultaneously visible.
        """
    )


def _render_analysis_dashboard(
    pose: PoseAnalysisResult,
    metadata: VideoMetadata,
    original_name: str,
    exercise_type: str,
    requested_side: str,
    min_visibility: float,
    smoothing_window: int,
) -> None:
    st.divider()
    st.header("Movement analysis dashboard")
    if pose.detected_frame_count == 0:
        st.error(
            "No pose was detected. Use a video with one fully visible person, "
            "good lighting and a stable camera."
        )
        return

    landmarks = pose.landmarks_dataframe()
    kinematics = compute_kinematics(
        landmarks,
        frame_width=pose.frame_width,
        frame_height=pose.frame_height,
        min_visibility=min_visibility,
        requested_side=requested_side,
        smoothing_window=smoothing_window,
    )
    exercise = _exercise_key(exercise_type)
    detection = detect_repetitions(
        kinematics.data["knee_angle_degrees"],
        kinematics.data["timestamp_seconds"],
        exercise_type=exercise,
    )
    repetition_metrics, repetition_summary = compute_repetition_metrics(
        kinematics.data,
        detection.repetitions,
        selected_side=kinematics.selected_side,
        exercise_type=exercise,
    )

    _render_summary(
        pose, kinematics, repetition_summary, exercise_type, metadata
    )
    overview, angles, repetitions, data, limitations = st.tabs(
        ["Overview", "Kinematics", "Repetitions", "Data & exports", "Limitations"]
    )
    with overview:
        _render_overview(pose, kinematics, requested_side)
    with angles:
        _render_kinematics(kinematics, detection)
    with repetitions:
        _render_repetitions(detection, repetition_metrics, repetition_summary)
    with data:
        _render_data_exports(
            pose,
            kinematics,
            detection,
            repetition_metrics,
            repetition_summary,
            metadata,
            exercise_type,
            original_name,
        )
    with limitations:
        _render_limitations()


def render_video_upload(
    min_visibility: float,
    target_fps: float,
    analysis_side: str,
    smoothing_window: int,
    exercise_type: str,
) -> None:
    st.header("1. Upload and analyze")
    uploaded_file = st.file_uploader(
        "Choose a rehabilitation exercise video",
        type=list(ALLOWED_VIDEO_TYPES),
        help="Accepted formats: MP4, MOV and AVI.",
    )
    if uploaded_file is None:
        st.info("Upload a video to display its preview and metadata.")
        return

    video_bytes = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_video:
            temp_video.write(video_bytes)
            temp_path = Path(temp_video.name)
        metadata = read_video_metadata(temp_path)

        video_column, metadata_column = st.columns((3, 2))
        with video_column:
            st.video(video_bytes)
            st.caption(uploaded_file.name)
        with metadata_column:
            _render_metadata(metadata)

        signature = _analysis_signature(video_bytes, min_visibility, target_fps)
        if st.button(
            "Run / refresh analysis",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Running MediaPipe pose estimation..."):
                st.session_state[SESSION_RESULT_KEY] = _analyze_uploaded_video(
                    video_bytes,
                    suffix,
                    min_visibility,
                    target_fps,
                )
                st.session_state[SESSION_SIGNATURE_KEY] = signature

        saved_result = st.session_state.get(SESSION_RESULT_KEY)
        saved_signature = st.session_state.get(SESSION_SIGNATURE_KEY)
        if saved_result is not None and saved_signature == signature:
            _render_analysis_dashboard(
                saved_result,
                metadata,
                uploaded_file.name,
                exercise_type,
                analysis_side,
                min_visibility,
                smoothing_window,
            )
        elif saved_result is not None:
            st.info(
                "Pose-processing settings changed. Click **Run / refresh "
                "analysis** to update the dashboard."
            )
        else:
            st.info("Click **Run / refresh analysis** to generate the dashboard.")
    except InvalidVideoError as exc:
        st.error(str(exc))
    except (PoseAnalysisError, PoseModelError, ValueError) as exc:
        st.error(str(exc))
    except OSError:
        st.error("The video could not be processed. Please try another file.")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

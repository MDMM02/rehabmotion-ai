from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rehabmotion.analysis.movement_metrics import compute_repetition_metrics
from rehabmotion.analysis.repetition_detection import (
    RepetitionDetectionResult,
    RepetitionSegment,
    detect_repetitions,
)
from rehabmotion.biomechanics.kinematics import KinematicsResult, compute_kinematics
from rehabmotion.pose.landmarks import POSE_CONNECTIONS
from rehabmotion.video.reader import read_video_metadata
from rehabmotion.video.writer import write_bgr_video


CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
PANEL_BACKGROUND = (35, 31, 24)
INK = (229, 245, 239)
MUTED = (174, 194, 187)
GREEN = (104, 190, 97)
TEAL = (139, 163, 56)
ORANGE = (68, 139, 226)
CARD = (53, 48, 39)
LINE = (83, 76, 62)


def _is_finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = INK,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _active_repetition(
    repetitions: tuple[RepetitionSegment, ...], timestamp: float
) -> RepetitionSegment | None:
    for repetition in repetitions:
        if repetition.start_time_seconds <= timestamp <= repetition.end_time_seconds:
            return repetition
    return None


def _phase_label(
    repetition: RepetitionSegment | None,
    timestamp: float,
    exercise_type: str,
) -> str:
    if repetition is None:
        return "Transition"
    if exercise_type == "sit-to-stand":
        return "Rise" if timestamp <= repetition.turning_time_seconds else "Sit down"
    return "Descent" if timestamp <= repetition.turning_time_seconds else "Ascent"


def _draw_skeleton(
    image: np.ndarray,
    landmarks: pd.DataFrame,
    min_visibility: float,
    selected_side: str,
    knee_angle: float,
) -> None:
    height, width = image.shape[:2]
    points: dict[int, tuple[int, int, float]] = {}
    for item in landmarks.itertuples(index=False):
        points[int(item.landmark_id)] = (
            int(round(float(item.x) * width)),
            int(round(float(item.y) * height)),
            float(item.visibility),
        )

    for start, end in POSE_CONNECTIONS:
        if start not in points or end not in points:
            continue
        start_x, start_y, start_visibility = points[start]
        end_x, end_y, end_visibility = points[end]
        if min(start_visibility, end_visibility) < 0.20:
            continue
        cv2.line(
            image,
            (start_x, start_y),
            (end_x, end_y),
            TEAL,
            3,
            cv2.LINE_AA,
        )

    for landmark_id, (x, y, visibility) in points.items():
        if visibility < 0.20:
            continue
        color = GREEN if visibility >= min_visibility else ORANGE
        radius = 5 if landmark_id in {11, 12, 23, 24, 25, 26, 27, 28} else 3
        cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)

    knee_id = 25 if selected_side == "left" else 26
    if knee_id in points and _is_finite(knee_angle):
        x, y, _ = points[knee_id]
        label = f"{float(knee_angle):.0f} deg"
        cv2.rectangle(image, (x + 8, y - 30), (x + 82, y - 7), PANEL_BACKGROUND, -1)
        _put_text(image, label, (x + 14, y - 13), 0.46, INK, 1)


def _metric_card(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    label: str,
    value: str,
) -> None:
    cv2.rectangle(canvas, (x, y), (x + width, y + 67), CARD, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + 67), LINE, 1)
    _put_text(canvas, label.upper(), (x + 13, y + 22), 0.34, MUTED, 1)
    _put_text(canvas, value, (x + 13, y + 51), 0.66, INK, 2)


def _draw_sparkline(
    canvas: np.ndarray,
    dataframe: pd.DataFrame,
    timestamp: float,
    start_time: float,
    end_time: float,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    cv2.rectangle(canvas, (x, y), (x + width, y + height), CARD, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), LINE, 1)
    _put_text(canvas, "KNEE ANGLE", (x + 12, y + 20), 0.34, MUTED, 1)
    chart_left, chart_top = x + 12, y + 29
    chart_width, chart_height = width - 24, height - 41
    window = dataframe[
        dataframe["timestamp_seconds"].between(start_time, end_time)
    ]
    finite = window[
        np.isfinite(pd.to_numeric(window["knee_angle_degrees"], errors="coerce"))
    ]
    if finite.empty or end_time <= start_time:
        return

    times = finite["timestamp_seconds"].to_numpy(dtype=float)
    angles = finite["knee_angle_degrees"].to_numpy(dtype=float)
    low = float(np.nanmin(angles)) - 5.0
    high = float(np.nanmax(angles)) + 5.0
    span = max(1.0, high - low)
    points = np.column_stack(
        (
            chart_left + (times - start_time) / (end_time - start_time) * chart_width,
            chart_top + (high - angles) / span * chart_height,
        )
    ).astype(np.int32)
    if len(points) > 1:
        cv2.polylines(canvas, [points], False, GREEN, 2, cv2.LINE_AA)
    current_x = int(
        round(chart_left + (timestamp - start_time) / (end_time - start_time) * chart_width)
    )
    current_x = max(chart_left, min(chart_left + chart_width, current_x))
    cv2.line(
        canvas,
        (current_x, chart_top),
        (current_x, chart_top + chart_height),
        ORANGE,
        2,
        cv2.LINE_AA,
    )


def _render_frame(
    source_bgr: np.ndarray,
    landmarks: pd.DataFrame,
    kinematics_row: pd.Series,
    kinematics: KinematicsResult,
    detection: RepetitionDetectionResult,
    timestamp: float,
    exercise_type: str,
    tempo_regularity: str,
    mean_rep_duration: float,
    min_visibility: float,
    demo_start: float,
    demo_end: float,
    credit: str,
) -> np.ndarray:
    source_height, source_width = source_bgr.shape[:2]
    video_width = int(round(CANVAS_HEIGHT * source_width / source_height))
    video = cv2.resize(source_bgr, (video_width, CANVAS_HEIGHT), interpolation=cv2.INTER_AREA)
    knee_angle = float(kinematics_row.get("knee_angle_degrees", float("nan")))
    _draw_skeleton(
        video,
        landmarks,
        min_visibility=min_visibility,
        selected_side=kinematics.selected_side,
        knee_angle=knee_angle,
    )

    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), PANEL_BACKGROUND, dtype=np.uint8)
    canvas[:, :video_width] = video
    cv2.rectangle(canvas, (0, 0), (video_width, 56), (20, 20, 20), -1)
    _put_text(canvas, "POSE + 2D KINEMATICS", (16, 25), 0.48, INK, 1)
    _put_text(canvas, f"t = {timestamp:.1f} s", (16, 46), 0.40, MUTED, 1)
    cv2.rectangle(
        canvas,
        (0, CANVAS_HEIGHT - 38),
        (video_width, CANVAS_HEIGHT),
        (20, 20, 20),
        -1,
    )
    _put_text(canvas, credit, (12, CANVAS_HEIGHT - 14), 0.30, MUTED, 1)

    panel_x = video_width + 27
    panel_width = CANVAS_WIDTH - panel_x - 25
    _put_text(canvas, "REHABMOTION AI", (panel_x, 48), 0.72, GREEN, 2)
    _put_text(
        canvas,
        f"{exercise_type.upper()}  |  {kinematics.selected_side.upper()} SIDE",
        (panel_x, 76),
        0.42,
        MUTED,
        1,
    )
    cv2.line(canvas, (panel_x, 95), (panel_x + panel_width, 95), LINE, 1)

    active = _active_repetition(detection.repetitions, timestamp)
    rep_label = (
        f"Rep {active.rep_id} / {len(detection.repetitions)}"
        if active is not None
        else f"{len(detection.repetitions)} reps detected"
    )
    _put_text(canvas, rep_label, (panel_x, 139), 0.95, INK, 2)
    _put_text(
        canvas,
        f"Phase: {_phase_label(active, timestamp, exercise_type)}",
        (panel_x, 172),
        0.52,
        ORANGE,
        1,
    )

    gap = 12
    card_width = (panel_width - gap) // 2
    _metric_card(
        canvas,
        panel_x,
        196,
        card_width,
        "Knee angle",
        f"{knee_angle:.0f} deg" if _is_finite(knee_angle) else "N/A",
    )
    _metric_card(
        canvas,
        panel_x + card_width + gap,
        196,
        card_width,
        "Knee ROM",
        f"{kinematics.metrics.knee_rom_degrees:.0f} deg",
    )
    active_duration = active.duration_seconds if active is not None else mean_rep_duration
    _metric_card(
        canvas,
        panel_x,
        275,
        card_width,
        "Rep duration",
        f"{active_duration:.2f} s" if _is_finite(active_duration) else "N/A",
    )
    _metric_card(
        canvas,
        panel_x + card_width + gap,
        275,
        card_width,
        "Tempo",
        tempo_regularity.title(),
    )
    _draw_sparkline(
        canvas,
        kinematics.data,
        timestamp,
        demo_start,
        demo_end,
        panel_x,
        365,
        panel_width,
        128,
    )
    _put_text(
        canvas,
        "Educational R&D prototype",
        (panel_x, 540),
        0.38,
        MUTED,
        1,
    )
    _put_text(
        canvas,
        "Not a medical device",
        (panel_x, 563),
        0.38,
        ORANGE,
        1,
    )
    return canvas


def build_portfolio_assets(
    *,
    video_path: Path,
    landmarks_path: Path,
    output_dir: Path,
    exercise_type: str = "squat",
    requested_side: str = "auto",
    min_visibility: float = 0.6,
    output_fps: float = 5.0,
    repetition_count: int = 2,
    start_repetition: int = 1,
    credit: str = "Demo source: tixonov_valentin via Pixabay #178381",
) -> tuple[Path, Path]:
    """Create a compact annotated MP4 and GIF for the project portfolio."""
    if output_fps <= 0:
        raise ValueError("output_fps must be greater than zero")
    if repetition_count <= 0 or start_repetition <= 0:
        raise ValueError("repetition selections must be positive")
    metadata = read_video_metadata(video_path)
    landmarks = pd.read_csv(landmarks_path)
    kinematics = compute_kinematics(
        landmarks,
        frame_width=metadata.width,
        frame_height=metadata.height,
        min_visibility=min_visibility,
        requested_side=requested_side,
        smoothing_window=7,
    )
    detection = detect_repetitions(
        kinematics.data["knee_angle_degrees"],
        kinematics.data["timestamp_seconds"],
        exercise_type=exercise_type,
    )
    repetition_metrics, summary = compute_repetition_metrics(
        kinematics.data,
        detection.repetitions,
        selected_side=kinematics.selected_side,
        exercise_type=exercise_type,
    )
    del repetition_metrics
    selected = detection.repetitions[
        start_repetition - 1 : start_repetition - 1 + repetition_count
    ]
    if not selected:
        raise ValueError("the requested repetitions were not detected")
    demo_start = max(0.0, selected[0].start_time_seconds - 0.25)
    demo_end = min(metadata.duration_seconds, selected[-1].end_time_seconds + 0.25)

    sampled = landmarks[["frame", "timestamp_seconds"]].drop_duplicates()
    sampled = sampled[sampled["timestamp_seconds"].between(demo_start, demo_end)]
    if sampled.empty:
        raise ValueError("no sampled landmark frames exist in the demo window")
    sampled_rate = 1.0 / float(sampled["timestamp_seconds"].diff().median())
    step = max(1, int(round(sampled_rate / output_fps)))
    sampled = sampled.iloc[::step]

    landmarks_by_frame = {
        int(frame): group
        for frame, group in landmarks.groupby("frame", sort=False)
    }
    kinematics_by_frame = kinematics.data.set_index("frame")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open demo video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        for sample in sampled.itertuples(index=False):
            frame_number = int(sample.frame)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, source_frame = capture.read()
            if not success or frame_number not in kinematics_by_frame.index:
                continue
            frames.append(
                _render_frame(
                    source_frame,
                    landmarks_by_frame[frame_number],
                    kinematics_by_frame.loc[frame_number],
                    kinematics,
                    detection,
                    float(sample.timestamp_seconds),
                    exercise_type,
                    summary.tempo_regularity,
                    summary.mean_duration_seconds,
                    min_visibility,
                    demo_start,
                    demo_end,
                    credit,
                )
            )
    finally:
        capture.release()
    if not frames:
        raise ValueError("no demo frames could be rendered")

    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = write_bgr_video(
        frames, output_dir / "rehabmotion_demo.mp4", fps=output_fps
    )
    gif_path = output_dir / "rehabmotion_demo.gif"
    gif_frames = [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
            (640, 480), Image.Resampling.LANCZOS
        )
        for frame in frames
    ]
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(round(1000 / output_fps)),
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(
        f"Created {len(frames)} frames | {len(detection.repetitions)} reps | "
        f"knee ROM {kinematics.metrics.knee_rom_degrees:.1f} deg | "
        f"tempo {summary.tempo_regularity}"
    )
    return mp4_path, gif_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the RehabMotion AI portfolio demo GIF and MP4."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument(
        "--exercise",
        choices=("squat", "sit-to-stand", "knee-flexion"),
        default="squat",
    )
    parser.add_argument("--side", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--start-repetition", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_portfolio_assets(
        video_path=args.video,
        landmarks_path=args.landmarks,
        output_dir=args.output_dir,
        exercise_type=args.exercise,
        requested_side=args.side,
        output_fps=args.fps,
        repetition_count=args.repetitions,
        start_repetition=args.start_repetition,
    )


if __name__ == "__main__":
    main()

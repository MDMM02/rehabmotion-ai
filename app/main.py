from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run app/main.py` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.upload import render_video_upload


DISCLAIMER = (
    "This prototype is for educational and R&D purposes only. "
    "It is not a certified medical device and should not be used for diagnosis, "
    "treatment decisions or clinical monitoring."
)


def render_future_sections() -> None:
    """Show completed and upcoming delivery phases."""
    st.divider()
    st.subheader("Development roadmap")
    st.progress(1.0, text="7 of 7 phases completed")

    completed = (
        ("Phase 1", "Video upload"),
        ("Phase 2", "Pose estimation"),
        ("Phase 3", "2D kinematics"),
        ("Phase 4", "Repetitions"),
        ("Phase 5", "Dashboard"),
        ("Phase 6", "PDF report"),
        ("Phase 7", "Portfolio"),
    )
    for start in range(0, len(completed), 3):
        group = completed[start : start + 3]
        row = st.columns(len(group))
        for column, (phase, label) in zip(row, group):
            column.success(f"**{phase}:** {label} — Complete")


def render_app_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 0.75rem;
            padding: 0.85rem 1rem;
        }
        [data-testid="stMetricLabel"] {
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="RehabMotion AI",
        page_icon="🦿",
        layout="wide",
    )
    render_app_styles()

    st.title("RehabMotion AI")
    st.caption("Video-based movement analysis for rehabilitation")
    st.markdown(
        "**Portfolio-ready release** · Pose → Kinematics → Repetitions → Reports"
    )
    st.warning(f"**Prototype R&D — not a medical device**\n\n{DISCLAIMER}")

    st.sidebar.header("Analysis setup")
    exercise = st.sidebar.selectbox(
        "Exercise",
        options=("Sit-to-stand", "Squat", "Knee flexion"),
        help="This controls the expected knee-angle cycle for repetition detection.",
    )
    min_visibility = st.sidebar.slider(
        "Minimum landmark visibility",
        min_value=0.30,
        max_value=0.90,
        value=0.60,
        step=0.05,
        help=(
            "Shoulders, hips, knees and ankles below this score mark a frame "
            "as low confidence."
        ),
    )
    target_fps = st.sidebar.select_slider(
        "Pose processing rate",
        options=(5, 10, 15),
        value=10,
        format_func=lambda value: f"{value} FPS",
        help="A lower rate is faster; 10 FPS is a good prototype default.",
    )
    analysis_side = st.sidebar.selectbox(
        "Analysis side",
        options=("Auto", "Left", "Right"),
        help="Auto selects the side with the best shoulder-to-ankle visibility.",
    )
    smoothing_window = st.sidebar.select_slider(
        "Smoothing window",
        options=(5, 7, 9, 11),
        value=7,
        help="Savitzky-Golay window in processed frames.",
    )

    render_video_upload(
        min_visibility=min_visibility,
        target_fps=float(target_fps),
        analysis_side=analysis_side,
        smoothing_window=smoothing_window,
        exercise_type=exercise,
    )
    render_future_sections()

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()

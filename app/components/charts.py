from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rehabmotion.analysis.repetition_detection import RepetitionSegment


def build_kinematics_figure(
    data: pd.DataFrame,
    selected_side: str,
    repetitions: Sequence[RepetitionSegment] = (),
) -> go.Figure:
    """Build raw-versus-smoothed knee, hip and trunk angle charts."""
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Knee angle", "Hip angle", "Trunk lean"),
    )
    signals = (
        ("knee_angle", "Knee", 1),
        ("hip_angle", "Hip", 2),
        ("trunk_lean", "Trunk", 3),
    )
    for signal, label, row in signals:
        figure.add_trace(
            go.Scatter(
                x=data["timestamp_seconds"],
                y=data[f"{signal}_raw_degrees"],
                name=f"{label} raw",
                mode="lines",
                line={"color": "rgba(120, 130, 145, 0.45)", "width": 1},
                connectgaps=False,
                legendgroup="raw",
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=data["timestamp_seconds"],
                y=data[f"{signal}_degrees"],
                name=f"{label} smoothed",
                mode="lines",
                line={"color": "#22c55e", "width": 2.5},
                connectgaps=False,
                legendgroup="smoothed",
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(title_text="Degrees", row=row, col=1)

    for repetition in repetitions:
        figure.add_vrect(
            x0=repetition.start_time_seconds,
            x1=repetition.end_time_seconds,
            fillcolor="#38bdf8" if repetition.rep_id % 2 else "#a78bfa",
            opacity=0.08,
            line_width=0,
            row="all",
            col=1,
        )
        figure.add_annotation(
            x=repetition.turning_time_seconds,
            y=repetition.knee_min_degrees,
            text=f"Rep {repetition.rep_id}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-25,
            row=1,
            col=1,
        )

    figure.update_xaxes(title_text="Time (s)", row=3, col=1)
    figure.update_layout(
        title=f"2D kinematics — {selected_side.title()} side",
        height=720,
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        hovermode="x unified",
    )
    return figure

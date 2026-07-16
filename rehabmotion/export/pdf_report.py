from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
import math
import os
from pathlib import Path
import tempfile

_matplotlib_cache = Path(tempfile.gettempdir()) / "rehabmotion-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from rehabmotion.analysis.report_data import MovementReportData


DISCLAIMER = (
    "This prototype is for educational and R&D purposes only. It is not a "
    "certified medical device and should not be used for diagnosis, treatment "
    "decisions or clinical monitoring."
)

INK = colors.HexColor("#17332D")
GREEN = colors.HexColor("#2F7D68")
GREEN_LIGHT = colors.HexColor("#EAF4F0")
TEAL = colors.HexColor("#38A38B")
ORANGE = colors.HexColor("#E28B44")
ORANGE_LIGHT = colors.HexColor("#FFF3E8")
SLATE = colors.HexColor("#52645F")
LINE = colors.HexColor("#D7E2DE")
PAPER = colors.HexColor("#F7FAF9")


def _is_finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _format_number(value: float, suffix: str = "", digits: int = 1) -> str:
    if not _is_finite(value):
        return "N/A"
    return f"{float(value):.{digits}f}{suffix}"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=29,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=4 * mm,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=SLATE,
        ),
        "section": ParagraphStyle(
            "ReportSection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=INK,
            spaceBefore=3 * mm,
            spaceAfter=3 * mm,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=SLATE,
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=SLATE,
        ),
        "warning": ParagraphStyle(
            "ReportWarning",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#824515"),
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=9,
            textColor=SLATE,
            alignment=TA_CENTER,
            spaceAfter=1.5 * mm,
        ),
        "metric_value": ParagraphStyle(
            "MetricValue",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=INK,
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=9,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9,
            textColor=INK,
            alignment=TA_CENTER,
        ),
    }


def _metric_card(label: str, value: str, styles: dict[str, ParagraphStyle]) -> Table:
    card = Table(
        [[Paragraph(escape(label), styles["metric_label"])],
         [Paragraph(escape(value), styles["metric_value"])]],
        colWidths=[54 * mm],
        rowHeights=[9 * mm, 12 * mm],
    )
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PAPER),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5 * mm),
            ]
        )
    )
    return card


def _metric_grid(
    metrics: list[tuple[str, str]], styles: dict[str, ParagraphStyle]
) -> Table:
    rows = []
    for index in range(0, len(metrics), 3):
        row = [
            _metric_card(label, value, styles)
            for label, value in metrics[index : index + 3]
        ]
        while len(row) < 3:
            row.append(Spacer(54 * mm, 1))
        rows.append(row)
    grid = Table(rows, colWidths=[58 * mm] * 3, hAlign="LEFT")
    grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
            ]
        )
    )
    return grid


def _info_table(rows: list[tuple[str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    data = [
        [
            Paragraph(f"<b>{escape(label)}</b>", styles["body"]),
            Paragraph(escape(value), styles["body"]),
        ]
        for label, value in rows
    ]
    table = Table(data, colWidths=[45 * mm, 125 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), GREEN_LIGHT),
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2.2 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2 * mm),
            ]
        )
    )
    return table


def _build_kinematics_chart(report: MovementReportData) -> BytesIO:
    dataframe = report.kinematics
    figure, axes = plt.subplots(3, 1, figsize=(7.3, 5.2), sharex=True)
    series = (
        ("knee_angle_degrees", "Knee angle", "#2F7D68"),
        ("hip_angle_degrees", "Hip angle", "#3978A8"),
        ("trunk_lean_degrees", "Trunk lean", "#E28B44"),
    )
    timestamps = pd.to_numeric(
        dataframe.get("timestamp_seconds", pd.Series(dtype=float)), errors="coerce"
    ).to_numpy(dtype=float)

    for axis, (column, label, color) in zip(axes, series):
        values = pd.to_numeric(
            dataframe.get(column, pd.Series(dtype=float)), errors="coerce"
        ).to_numpy(dtype=float)
        if len(timestamps) == len(values) and np.isfinite(values).any():
            axis.plot(timestamps, values, color=color, linewidth=1.8)
        else:
            axis.text(
                0.5,
                0.5,
                "No reliable samples",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#52645F",
            )
        for _, repetition in report.repetitions.iterrows():
            start = repetition.get("start_time_seconds")
            end = repetition.get("end_time_seconds")
            if _is_finite(start) and _is_finite(end):
                axis.axvspan(float(start), float(end), color="#38A38B", alpha=0.09)
        axis.set_ylabel("deg", fontsize=8)
        axis.set_title(label, loc="left", fontsize=9, fontweight="bold")
        axis.grid(axis="y", color="#D7E2DE", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7, colors="#52645F")

    axes[-1].set_xlabel("Time (seconds)", fontsize=8)
    figure.suptitle(
        f"Smoothed 2D kinematics - {report.analyzed_side.title()} side",
        x=0.09,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color="#17332D",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output = BytesIO()
    figure.savefig(output, format="png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    output.seek(0)
    return output


def _phase_columns(dataframe: pd.DataFrame) -> list[str]:
    excluded = {
        "duration_seconds",
        "start_time_seconds",
        "turning_time_seconds",
        "end_time_seconds",
    }
    return [
        column
        for column in dataframe.columns
        if column.endswith("_duration_seconds") and column not in excluded
    ][:2]


def _repetition_table(
    dataframe: pd.DataFrame, styles: dict[str, ParagraphStyle]
) -> Table | Paragraph:
    if dataframe.empty:
        return Paragraph(
            "No complete repetition was detected for this analysis.", styles["body"]
        )

    phase_columns = _phase_columns(dataframe)
    columns: list[tuple[str, str, str]] = [("rep", "Rep", "int")]
    columns.append(("duration_seconds", "Duration (s)", "float"))
    for column in phase_columns:
        label = column.removesuffix("_duration_seconds").replace("_", " ").title()
        columns.append((column, f"{label} (s)", "float"))
    columns.extend(
        [
            ("knee_rom_degrees", "Knee ROM (deg)", "float"),
            ("hip_rom_degrees", "Hip ROM (deg)", "float"),
            ("max_trunk_lean_degrees", "Max trunk (deg)", "float"),
            ("mean_knee_asymmetry_degrees", "Mean asym. (deg)", "float"),
        ]
    )
    header = [Paragraph(escape(label), styles["table_header"]) for _, label, _ in columns]
    rows = [header]
    for _, item in dataframe.iterrows():
        row = []
        for column, _, kind in columns:
            value = item.get(column)
            formatted = (
                str(int(value))
                if kind == "int" and _is_finite(value)
                else _format_number(value, digits=2)
            )
            row.append(Paragraph(escape(formatted), styles["table_cell"]))
        rows.append(row)

    available_width = 170 * mm
    first_width = 12 * mm
    remaining_width = (available_width - first_width) / (len(columns) - 1)
    table = Table(
        rows,
        colWidths=[first_width] + [remaining_width] * (len(columns) - 1),
        repeatRows=1,
        hAlign="LEFT",
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), GREEN),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.5 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.5 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
    ]
    for row_index in range(1, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PAPER))
    table.setStyle(TableStyle(commands))
    return table


def _draw_header_footer(canvas, document) -> None:
    canvas.saveState()
    page_width, page_height = A4
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(20 * mm, page_height - 15 * mm, page_width - 20 * mm, page_height - 15 * mm)
    canvas.setFillColor(GREEN)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(20 * mm, page_height - 11 * mm, "REHABMOTION AI")
    canvas.setFillColor(SLATE)
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(page_width - 20 * mm, page_height - 11 * mm, "MOVEMENT ANALYSIS REPORT")

    canvas.line(20 * mm, 15 * mm, page_width - 20 * mm, 15 * mm)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawString(20 * mm, 10.5 * mm, "Educational R&D prototype - not a medical device")
    canvas.drawRightString(
        page_width - 20 * mm, 10.5 * mm, f"Page {document.page}"
    )
    canvas.restoreState()


def generate_pdf_report(
    report: MovementReportData,
    *,
    generated_at: datetime | None = None,
) -> bytes:
    """Generate a polished, self-contained movement-analysis PDF report."""
    styles = _styles()
    generated = generated_at or datetime.now().astimezone()
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=23 * mm,
        bottomMargin=22 * mm,
        title="RehabMotion AI movement analysis report",
        author="RehabMotion AI",
        subject="Educational video-based movement analysis",
    )
    story = []

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Movement analysis report", styles["title"]))
    story.append(
        Paragraph(
            "RehabMotion AI | Video-based movement analysis for rehabilitation",
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 6 * mm))
    generated_text = generated.strftime("%Y-%m-%d %H:%M %Z").strip()
    identity = _info_table(
        [
            ("Source file", report.source_name),
            ("Exercise", report.exercise_type),
            ("Analyzed side", report.analyzed_side.title()),
            ("Generated", generated_text),
        ],
        styles,
    )
    story.append(identity)
    story.append(Spacer(1, 4 * mm))
    warning_box = Table(
        [[Paragraph(escape(DISCLAIMER), styles["warning"])]],
        colWidths=[170 * mm],
    )
    warning_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), ORANGE_LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.8, ORANGE),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )
    story.append(warning_box)
    story.append(Paragraph("Analysis overview", styles["section"]))
    movement = report.movement
    overview_metrics = [
        ("Detected repetitions", str(movement.detected_repetitions)),
        ("Knee ROM", _format_number(movement.knee_rom_degrees, " deg")),
        ("Hip ROM", _format_number(movement.hip_rom_degrees, " deg")),
        (
            "Mean rep duration",
            _format_number(movement.mean_repetition_duration_seconds, " s", 2),
        ),
        ("Max trunk lean", _format_number(movement.trunk_max_degrees, " deg")),
        ("Tempo regularity", movement.tempo_regularity.title()),
    ]
    story.append(_metric_grid(overview_metrics, styles))

    story.append(Paragraph("Video and data quality", styles["section"]))
    video = report.video
    quality = report.quality
    story.append(
        _info_table(
            [
                ("Video duration", _format_number(video.duration_seconds, " s", 2)),
                ("Video rate", _format_number(video.fps, " FPS", 2)),
                ("Frame count", f"{video.frame_count:,}"),
                ("Resolution", f"{video.width} x {video.height}"),
                ("Pose detected", _format_number(quality.pose_detection_rate * 100, "%")),
                (
                    "Bilateral reliability",
                    _format_number(quality.bilateral_reliable_rate * 100, "%"),
                ),
                ("Mean visibility", _format_number(quality.mean_visibility, digits=2)),
                (
                    "Usable kinematics",
                    _format_number(quality.usable_kinematics_rate * 100, "%"),
                ),
            ],
            styles,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Joint-angle curves", styles["section"]))
    story.append(
        Paragraph(
            "Smoothed 2D estimates. Shaded intervals mark complete detected repetitions. "
            "Missing or low-confidence samples are not interpolated across long gaps.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 3 * mm))
    chart = _build_kinematics_chart(report)
    story.append(Image(chart, width=170 * mm, height=121 * mm))
    story.append(Paragraph("Kinematic summary", styles["section"]))
    story.append(
        _info_table(
            [
                ("Knee angle range", f"{_format_number(movement.knee_min_degrees)} to {_format_number(movement.knee_max_degrees)} deg"),
                ("Knee ROM", _format_number(movement.knee_rom_degrees, " deg")),
                ("Hip angle range", f"{_format_number(movement.hip_min_degrees)} to {_format_number(movement.hip_max_degrees)} deg"),
                ("Hip ROM", _format_number(movement.hip_rom_degrees, " deg")),
                ("Mean trunk lean", _format_number(movement.trunk_mean_degrees, " deg")),
                ("Max trunk lean", _format_number(movement.trunk_max_degrees, " deg")),
                ("Mean knee asymmetry", _format_number(movement.knee_asymmetry_mean_degrees, " deg")),
                ("Max knee asymmetry", _format_number(movement.knee_asymmetry_max_degrees, " deg")),
            ],
            styles,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Repetition analysis", styles["section"]))
    story.append(
        Paragraph(
            "Only complete cycles crossing both adaptive thresholds are counted. "
            "Partial movements are ignored.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(_repetition_table(report.repetitions, styles))
    story.append(Spacer(1, 5 * mm))
    detection_rows = [
        ("Signal excursion", _format_number(report.signal_excursion_degrees, " deg")),
        ("Start / return threshold", _format_number(report.start_threshold_degrees, " deg")),
        ("Turning threshold", _format_number(report.turning_threshold_degrees, " deg")),
        ("Duration variability", _format_number(movement.duration_cv * 100, "%", 1)),
    ]
    story.append(KeepTogether([Paragraph("Detection details", styles["section"]), _info_table(detection_rows, styles)]))
    if report.detection_warning:
        story.append(Spacer(1, 3 * mm))
        warning = Table(
            [[Paragraph(escape(report.detection_warning), styles["warning"])]],
            colWidths=[170 * mm],
        )
        warning.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), ORANGE_LIGHT),
                    ("BOX", (0, 0), (-1, -1), 0.6, ORANGE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5 * mm),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ]
            )
        )
        story.append(warning)

    story.append(Paragraph("Interpretation and limitations", styles["section"]))
    limitations = (
        "- Results are single-camera 2D estimates and depend on camera placement.<br/>"
        "- Occlusion, loose clothing, lighting and out-of-plane motion reduce accuracy.<br/>"
        "- ROM values are not equivalent to clinical goniometry or 3D motion capture.<br/>"
        "- Tempo and repetition thresholds are heuristics requiring visual review.<br/>"
        "- Left-right asymmetry requires both sides to be visible at the same time."
    )
    story.append(Paragraph(limitations, styles["body"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(f"<b>Medical disclaimer:</b> {escape(DISCLAIMER)}", styles["small"]))

    document.build(
        story,
        onFirstPage=_draw_header_footer,
        onLaterPages=_draw_header_footer,
    )
    return output.getvalue()

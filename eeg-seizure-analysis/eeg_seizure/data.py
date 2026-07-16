"""Dataset loading, validation and recording-level grouping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Epileptic Seizure Recognition.csv"
SAMPLING_RATE_HZ = 173.61
EXPECTED_SIGNAL_LENGTH = 178

CLASS_LABELS = {
    1: "Crise (ictal)",
    2: "Zone épileptogène, hors crise",
    3: "Hippocampe opposé, hors crise",
    4: "Sujet sain, yeux fermés",
    5: "Sujet sain, yeux ouverts",
}


def signal_columns(frame: pd.DataFrame) -> list[str]:
    """Return X1...X178 in numeric order."""
    columns = [column for column in frame.columns if column.startswith("X") and column[1:].isdigit()]
    return sorted(columns, key=lambda column: int(column[1:]))


def _recording_groups(identifiers: pd.Series, labels: pd.Series) -> pd.Series:
    """Recover the 500 parent recordings encoded in the identifier column.

    The first token (X1...X23) is the chunk index and the last token identifies
    the parent recording. Five source identifiers omit the last token, one per
    class, so the class is used to distinguish those five groups.
    """
    suffix = identifiers.astype("string").str.split(".").str[2]
    groups = "recording_" + suffix.fillna("").str.zfill(3)
    missing = suffix.isna() | suffix.eq("")
    groups.loc[missing] = labels.loc[missing].map(lambda value: f"recording_missing_class_{int(value)}")
    return groups.astype("string")


def validate_dataset(frame: pd.DataFrame) -> None:
    """Raise a useful error when a file is not the expected Kaggle dataset."""
    columns = signal_columns(frame)
    if len(columns) != EXPECTED_SIGNAL_LENGTH:
        raise ValueError(f"178 colonnes EEG attendues, {len(columns)} trouvées.")
    if "y" not in frame.columns:
        raise ValueError("La colonne cible 'y' est absente.")
    if set(frame["y"].dropna().astype(int).unique()) != set(CLASS_LABELS):
        raise ValueError("Les classes attendues sont 1, 2, 3, 4 et 5.")
    if frame[columns + ["y"]].isna().any().any():
        raise ValueError("Le dataset contient des valeurs manquantes dans les signaux ou la cible.")


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the Kaggle CSV and add analysis metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}. Lancez `python scripts/download_data.py`."
        )

    frame = pd.read_csv(path)
    validate_dataset(frame)
    identifier_column = "Unnamed" if "Unnamed" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={identifier_column: "segment_id"}).copy()
    frame["y"] = frame["y"].astype(int)
    metadata = pd.DataFrame(
        {
            "is_seizure": (frame["y"] == 1).astype(int),
            "class_name": frame["y"].map(CLASS_LABELS),
            "recording_group": _recording_groups(frame["segment_id"], frame["y"]),
        },
        index=frame.index,
    )
    return pd.concat([frame, metadata], axis=1).copy()


def dataset_summary(frame: pd.DataFrame) -> dict[str, int | float]:
    """Small serializable dataset quality summary."""
    columns = signal_columns(frame)
    return {
        "n_segments": int(len(frame)),
        "n_parent_recordings": int(frame["recording_group"].nunique()),
        "n_signal_points": int(len(columns)),
        "n_classes": int(frame["y"].nunique()),
        "n_seizure_segments": int(frame["is_seizure"].sum()),
        "seizure_prevalence": float(frame["is_seizure"].mean()),
        "missing_values": int(frame[columns + ["y"]].isna().sum().sum()),
        "duplicate_signals": int(frame[columns].duplicated().sum()),
    }

from __future__ import annotations

import pandas as pd


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a dataframe as a UTF-8 CSV payload for downloads."""
    return dataframe.to_csv(index=False, lineterminator="\n").encode("utf-8")

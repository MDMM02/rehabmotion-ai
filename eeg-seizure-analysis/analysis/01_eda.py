# %% [markdown]
# # EDA — Epileptic Seizure Recognition
# Script à cellules, ouvrable comme notebook dans VS Code/Jupyter.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# %%
from eeg_seizure.data import dataset_summary, load_dataset, signal_columns
from eeg_seizure.features import feature_table

data = load_dataset()
dataset_summary(data)

# %%
data.groupby(["y", "class_name"]).agg(
    segments=("segment_id", "size"),
    acquisitions=("recording_group", "nunique"),
)

# %%
features = feature_table(data)
features.groupby("class_name")[["rms", "line_length", "kurtosis", "spectral_entropy"]].median()

# %% [markdown]
# Le rapport complet et les graphiques sont reproductibles avec :
# `python scripts/run_eda.py`

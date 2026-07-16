# %% [markdown]
# # ML — classification crise / hors crise
# Le split est réalisé au niveau des acquisitions parentes pour éviter la fuite
# entre les 23 fenêtres issues du même signal source.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# %%
from eeg_seizure.data import load_dataset
from eeg_seizure.features import feature_table
from eeg_seizure.modeling import train_candidates

data = load_dataset()
features = feature_table(data)
results = train_candidates(features)
results["metrics"]

# %%
results["predictions"].head()

# %% [markdown]
# Pour enregistrer le meilleur modèle, les métriques, les prédictions et les
# figures : `python scripts/train_model.py`

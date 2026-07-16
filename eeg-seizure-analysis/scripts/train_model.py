"""Train, compare and persist leakage-aware binary seizure classifiers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

sys.path.insert(0, str(ROOT))

from eeg_seizure.data import SAMPLING_RATE_HZ, load_dataset, signal_columns  # noqa: E402
from eeg_seizure.features import feature_table  # noqa: E402
from eeg_seizure.modeling import feature_importance, train_candidates  # noqa: E402


REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    raw = load_dataset()
    features = feature_table(raw)
    results = train_candidates(features)
    metrics = results["metrics"]
    predictions = results["predictions"]
    best_name = str(results["best_model_name"])
    model = results["model"]
    feature_names = list(results["feature_names"])

    metrics.to_csv(REPORTS / "model_comparison.csv", index=False)
    predictions.to_csv(REPORTS / "test_predictions.csv", index=False)
    importances = feature_importance(model, feature_names)
    importances.to_csv(REPORTS / "feature_importance.csv", index=False)

    best_metrics = metrics.loc[metrics["model"] == best_name].iloc[0].to_dict()
    serializable_metrics = {
        key: (float(value) if isinstance(value, (int, float)) and key != "model" else value)
        for key, value in best_metrics.items()
    }
    (REPORTS / "best_model_metrics.json").write_text(
        json.dumps(serializable_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    y_test = predictions["is_seizure"].to_numpy()
    probability = predictions["seizure_probability"].to_numpy()
    prediction = predictions["prediction"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ConfusionMatrixDisplay.from_predictions(y_test, prediction, display_labels=["Hors crise", "Crise"], cmap="Blues", ax=axes[0], colorbar=False)
    RocCurveDisplay.from_predictions(y_test, probability, ax=axes[1])
    PrecisionRecallDisplay.from_predictions(y_test, probability, ax=axes[2])
    axes[0].set_title("Matrice de confusion")
    axes[1].set_title("Courbe ROC")
    axes[2].set_title("Précision–rappel")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_evaluation.png", dpi=160)
    plt.close(fig)

    bundle = {
        "model": model,
        "model_name": best_name,
        "feature_names": feature_names,
        "signal_columns": signal_columns(raw),
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "threshold": 0.5,
        "metrics": serializable_metrics,
    }
    joblib.dump(bundle, ARTIFACTS / "seizure_model.joblib")

    report = f"""# Rapport ML — détection binaire de crise

## Protocole

- Cible : classe 1 contre classes 2 à 5.
- Variables : 28 caractéristiques temporelles et spectrales interprétables.
- Split : 80/20 stratifié au niveau des 500 acquisitions parentes (aucune fenêtre sœur entre train et test).
- Comparaison : Dummy, régression logistique, Random Forest et Extra Trees.
- Sélection : meilleure Average Precision en validation croisée groupée à 4 plis sur le train.
- Évaluation finale : holdout de 20 % jamais utilisé pour choisir le modèle, seuil fixé à 0,50.

## Meilleur modèle : `{best_name}`

- Average Precision en validation croisée : **{best_metrics['cv_average_precision']:.3f}**
- ROC AUC : **{best_metrics['roc_auc']:.3f}**
- Average Precision : **{best_metrics['average_precision']:.3f}**
- Sensibilité / rappel : **{best_metrics['recall_sensitivity']:.3f}**
- Spécificité : **{best_metrics['specificity']:.3f}**
- Précision : **{best_metrics['precision']:.3f}**
- F1 : **{best_metrics['f1']:.3f}**
- Balanced accuracy : **{best_metrics['balanced_accuracy']:.3f}**

## Lecture responsable

Ces scores mesurent la discrimination sur un benchmark nettoyé, pas la performance clinique sur de nouveaux patients. Les fenêtres partagent une provenance limitée à 500 acquisitions, et l'identité patient n'est pas fournie explicitement. Une validation externe, prospective et multi-centres serait nécessaire avant tout usage médical.
"""
    (REPORTS / "MODEL_REPORT.md").write_text(report, encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"\nArtefact enregistré : {ARTIFACTS / 'seizure_model.joblib'}")


if __name__ == "__main__":
    main()

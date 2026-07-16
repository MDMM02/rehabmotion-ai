"""Generate the reproducible EDA report and portfolio-ready figures."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(ROOT))

from eeg_seizure.data import (  # noqa: E402
    CLASS_LABELS,
    SAMPLING_RATE_HZ,
    dataset_summary,
    load_dataset,
    signal_columns,
)
from eeg_seizure.features import feature_table, model_feature_columns  # noqa: E402


REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
COLORS = {1: "#ef476f", 2: "#f78c6b", 3: "#ffd166", 4: "#06d6a0", 5: "#118ab2"}


def save_class_distribution(frame: pd.DataFrame) -> None:
    counts = frame["y"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar([CLASS_LABELS[index] for index in counts.index], counts, color=[COLORS[index] for index in counts.index])
    ax.bar_label(bars, padding=4)
    ax.set(title="Répartition équilibrée des cinq états EEG", ylabel="Nombre de segments")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(FIGURES / "class_distribution.png", dpi=160)
    plt.close(fig)


def save_mean_signals(frame: pd.DataFrame) -> None:
    columns = signal_columns(frame)
    time = np.arange(len(columns)) / SAMPLING_RATE_HZ
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    for label, axis in zip(sorted(CLASS_LABELS), axes):
        class_signals = frame.loc[frame["y"] == label, columns].to_numpy()
        mean = class_signals.mean(axis=0)
        lower, upper = np.percentile(class_signals, [25, 75], axis=0)
        axis.fill_between(time, lower, upper, color=COLORS[label], alpha=0.18, label="IQR")
        axis.plot(time, mean, color=COLORS[label], linewidth=1.3, label="Moyenne")
        axis.set_ylabel(f"Classe {label}")
        axis.grid(alpha=0.2)
    axes[0].set_title("Signal moyen et intervalle interquartile par classe")
    axes[-1].set_xlabel("Temps (s)")
    fig.tight_layout()
    fig.savefig(FIGURES / "mean_signals_by_class.png", dpi=160)
    plt.close(fig)


def save_feature_boxplots(features: pd.DataFrame) -> None:
    selected = ["rms", "line_length", "kurtosis", "beta_relative_power"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for feature, axis in zip(selected, axes.ravel()):
        values = [features.loc[features["y"] == label, feature].to_numpy() for label in CLASS_LABELS]
        axis.boxplot(values, tick_labels=[str(label) for label in CLASS_LABELS], showfliers=False, patch_artist=True)
        axis.set(title=feature.replace("_", " ").title(), xlabel="Classe")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Variables discriminantes : domaine temporel et fréquentiel", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_distributions.png", dpi=160)
    plt.close(fig)


def save_pca(features: pd.DataFrame) -> dict[str, float]:
    columns = model_feature_columns(features)
    scaled = StandardScaler().fit_transform(features[columns])
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled)
    fig, ax = plt.subplots(figsize=(10, 7))
    rng = np.random.default_rng(42)
    for label in CLASS_LABELS:
        indices = np.flatnonzero(features["y"].to_numpy() == label)
        if len(indices) > 900:
            indices = rng.choice(indices, 900, replace=False)
        ax.scatter(coordinates[indices, 0], coordinates[indices, 1], s=10, alpha=0.38, color=COLORS[label], label=f"{label} — {CLASS_LABELS[label]}")
    ax.set(
        title="Projection PCA des caractéristiques EEG",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
    )
    ax.legend(markerscale=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_pca.png", dpi=160)
    plt.close(fig)
    return {
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance": float(pca.explained_variance_ratio_[1]),
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = load_dataset()
    features = feature_table(frame)
    summary = dataset_summary(frame)

    class_summary = (
        features.groupby(["y", "class_name"])
        .agg(
            segments=("segment_id", "size"),
            recordings=("recording_group", "nunique"),
            rms_median=("rms", "median"),
            line_length_median=("line_length", "median"),
            dominant_frequency_median_hz=("dominant_frequency_hz", "median"),
            spectral_entropy_median=("spectral_entropy", "median"),
        )
        .reset_index()
    )
    class_summary.to_csv(REPORTS / "class_summary.csv", index=False)
    features.to_csv(REPORTS / "engineered_features.csv", index=False)

    save_class_distribution(frame)
    save_mean_signals(frame)
    save_feature_boxplots(features)
    pca_summary = save_pca(features)
    summary.update(pca_summary)
    (REPORTS / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report = f"""# Rapport EDA — EEG et crises épileptiques

## Qualité et structure

- **{summary['n_segments']:,} segments** de 178 points, sans valeur manquante.
- **{summary['n_parent_recordings']} acquisitions parentes** reconstruites, avec 23 fenêtres par acquisition.
- Cinq classes équilibrées à 2 300 segments chacune.
- La cible binaire « crise » représente **{summary['seizure_prevalence']:.0%}** des segments.
- {summary['duplicate_signals']} signaux dupliqués exactement.

## Premiers constats

La classe 1 (activité ictale) se distingue surtout par l'amplitude, l'énergie, la longueur de ligne et plusieurs caractéristiques spectrales. La PCA montre la structure globale mais aussi un recouvrement entre classes : une frontière non linéaire est donc pertinente. Les graphiques associés sont dans `reports/figures/` et le tableau des variables dans `reports/engineered_features.csv`.

## Point méthodologique critique

Les lignes ne sont pas indépendantes : chacune des 500 acquisitions originales a été découpée en 23 fenêtres. Toute évaluation ML doit conserver les 23 fenêtres d'une acquisition dans un seul split. Le pipeline de ce projet applique cette séparation par groupe.

## Limites

Ce benchmark monocanal, ancien, très nettoyé et composé de fenêtres courtes ne représente pas la diversité d'un EEG clinique continu. Il sert à démontrer une démarche data science, pas à valider un dispositif de diagnostic.
"""
    (REPORTS / "EDA_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

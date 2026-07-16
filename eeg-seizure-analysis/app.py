"""Interactive Streamlit portfolio app for EEG seizure exploration."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from eeg_seizure.data import (
    CLASS_LABELS,
    DEFAULT_DATA_PATH,
    SAMPLING_RATE_HZ,
    dataset_summary,
    load_dataset,
    signal_columns,
)
from eeg_seizure.features import EEG_BANDS, extract_features, feature_table


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "seizure_model.joblib"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="EEG Seizure Lab", page_icon="🧠", layout="wide")


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_dataset(DEFAULT_DATA_PATH)


@st.cache_data(show_spinner="Extraction des caractéristiques EEG…")
def get_features() -> pd.DataFrame:
    return feature_table(get_data())


@st.cache_resource(show_spinner=False)
def get_model() -> dict[str, object] | None:
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def signal_figure(row: pd.Series, columns: list[str]) -> go.Figure:
    time = np.arange(len(columns)) / SAMPLING_RATE_HZ
    figure = go.Figure(
        go.Scatter(x=time, y=row[columns], mode="lines", line={"color": "#7c3aed", "width": 2})
    )
    figure.update_layout(
        title=f"Segment {row['segment_id']} — {row['class_name']}",
        xaxis_title="Temps (s)",
        yaxis_title="Amplitude EEG (unité du dataset)",
        height=410,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    return figure


def spectrum_figure(row: pd.Series, columns: list[str]) -> go.Figure:
    signal = row[columns].to_numpy(dtype=float)
    frequency = np.fft.rfftfreq(len(signal), d=1 / SAMPLING_RATE_HZ)
    power = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    figure = go.Figure(go.Scatter(x=frequency, y=power, fill="tozeroy", line={"color": "#0ea5e9"}))
    for band, (low, high) in EEG_BANDS.items():
        figure.add_vrect(x0=low, x1=high, opacity=0.08, line_width=0, annotation_text=band)
    figure.update_layout(
        title="Spectre de puissance indicatif",
        xaxis_title="Fréquence (Hz)",
        yaxis_title="Puissance",
        xaxis_range=[0, 50],
        height=410,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    return figure


st.title("🧠 EEG Seizure Lab")
st.caption("Exploration de signaux EEG et classification binaire crise / hors crise — démonstrateur portfolio")
st.warning(
    "Projet éducatif uniquement. Ce modèle n’est ni un dispositif médical ni un outil de diagnostic, "
    "de surveillance ou de décision thérapeutique."
)

try:
    data = get_data()
except FileNotFoundError as error:
    st.error(str(error))
    st.code("python scripts/download_data.py", language="bash")
    st.stop()

columns = signal_columns(data)
summary = dataset_summary(data)
overview_tab, signal_tab, eda_tab, model_tab, method_tab = st.tabs(
    ["Vue d’ensemble", "Explorateur de signal", "EDA", "Modèle", "Méthode & limites"]
)

with overview_tab:
    left, middle_left, middle_right, right = st.columns(4)
    left.metric("Segments", f"{summary['n_segments']:,}".replace(",", " "))
    middle_left.metric("Acquisitions parentes", summary["n_parent_recordings"])
    middle_right.metric("Points / segment", summary["n_signal_points"])
    right.metric("Segments avec crise", f"{summary['seizure_prevalence']:.0%}")

    counts = data.groupby(["y", "class_name"], as_index=False).size()
    class_chart = px.bar(
        counts,
        x="class_name",
        y="size",
        color="class_name",
        text="size",
        labels={"class_name": "État EEG", "size": "Segments"},
        title="Répartition des classes",
    )
    class_chart.update_layout(showlegend=False, xaxis_tickangle=-15, height=450)
    st.plotly_chart(class_chart, width="stretch")
    st.info(
        "Les cinq classes sont équilibrées, mais la cible binaire ne l’est pas : la crise correspond à "
        "une classe sur cinq. L’Average Precision, le rappel et la spécificité complètent donc l’accuracy."
    )

with signal_tab:
    control_left, control_right = st.columns([2, 3])
    selected_label = control_left.selectbox(
        "Classe",
        options=list(CLASS_LABELS),
        format_func=lambda label: f"{label} — {CLASS_LABELS[label]}",
    )
    candidates = data.loc[data["y"] == selected_label]
    selected_index = control_right.slider("Exemple dans cette classe", 0, len(candidates) - 1, 0)
    row = candidates.iloc[selected_index]
    signal_left, signal_right = st.columns(2)
    signal_left.plotly_chart(signal_figure(row, columns), width="stretch")
    signal_right.plotly_chart(spectrum_figure(row, columns), width="stretch")

    sample_features = extract_features(row[columns].to_numpy()).iloc[0]
    st.subheader("Caractéristiques du segment")
    feature_cols = st.columns(5)
    feature_cols[0].metric("RMS", f"{sample_features['rms']:.1f}")
    feature_cols[1].metric("Amplitude crête-à-crête", f"{sample_features['peak_to_peak']:.0f}")
    feature_cols[2].metric("Longueur de ligne", f"{sample_features['line_length']:.0f}")
    feature_cols[3].metric("Fréquence dominante", f"{sample_features['dominant_frequency_hz']:.1f} Hz")
    feature_cols[4].metric("Entropie spectrale", f"{sample_features['spectral_entropy']:.2f}")

with eda_tab:
    features = get_features()
    selected_feature = st.selectbox(
        "Variable à comparer",
        options=[
            "rms",
            "peak_to_peak",
            "line_length",
            "kurtosis",
            "dominant_frequency_hz",
            "spectral_entropy",
            "delta_relative_power",
            "theta_relative_power",
            "alpha_relative_power",
            "beta_relative_power",
            "gamma_relative_power",
        ],
    )
    plot_frame = features[["class_name", selected_feature]].copy()
    lower, upper = plot_frame[selected_feature].quantile([0.01, 0.99])
    plot_frame = plot_frame[plot_frame[selected_feature].between(lower, upper)]
    violin = px.violin(
        plot_frame,
        x="class_name",
        y=selected_feature,
        color="class_name",
        box=True,
        points=False,
        title="Distribution par état EEG (valeurs entre les percentiles 1 et 99)",
    )
    violin.update_layout(showlegend=False, xaxis_tickangle=-15, height=520)
    st.plotly_chart(violin, width="stretch")

    summary_table = (
        features.groupby(["y", "class_name"])[selected_feature]
        .agg(["median", "mean", "std"])
        .round(3)
        .reset_index()
    )
    st.dataframe(summary_table, width="stretch", hide_index=True)

with model_tab:
    bundle = get_model()
    if bundle is None:
        st.info("Le modèle n’a pas encore été entraîné. Exécutez la commande suivante puis rechargez la page.")
        st.code("python scripts/train_model.py", language="bash")
    else:
        metrics = bundle["metrics"]
        st.subheader(f"Meilleur modèle : {bundle['model_name']}")
        metric_cols = st.columns(6)
        metric_cols[0].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        metric_cols[1].metric("Avg Precision", f"{metrics['average_precision']:.3f}")
        metric_cols[2].metric("Sensibilité", f"{metrics['recall_sensitivity']:.3f}")
        metric_cols[3].metric("Spécificité", f"{metrics['specificity']:.3f}")
        metric_cols[4].metric("Précision", f"{metrics['precision']:.3f}")
        metric_cols[5].metric("F1", f"{metrics['f1']:.3f}")

        evaluation_path = REPORTS / "figures" / "model_evaluation.png"
        if evaluation_path.exists():
            st.image(str(evaluation_path), caption="Évaluation sur le holdout par acquisition")

        st.subheader("Tester un segment du dataset")
        model_label = st.selectbox(
            "Classe réelle du segment",
            options=list(CLASS_LABELS),
            format_func=lambda label: f"{label} — {CLASS_LABELS[label]}",
            key="model_class",
        )
        model_candidates = data.loc[data["y"] == model_label]
        model_index = st.slider("Exemple", 0, len(model_candidates) - 1, 0, key="model_example")
        model_row = model_candidates.iloc[model_index]
        model_features = extract_features(model_row[columns].to_numpy())
        probability = float(bundle["model"].predict_proba(model_features[bundle["feature_names"]])[:, 1][0])
        prediction_text = "Crise détectée" if probability >= bundle["threshold"] else "Hors crise"
        prediction_color = "#ef476f" if probability >= bundle["threshold"] else "#06d6a0"
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={"text": f"{prediction_text} — score du modèle"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": prediction_color},
                    "steps": [{"range": [0, 50], "color": "#dcfce7"}, {"range": [50, 100], "color": "#fee2e2"}],
                    "threshold": {"line": {"color": "#111827", "width": 3}, "value": bundle["threshold"] * 100},
                },
            )
        )
        gauge.update_layout(height=330, margin={"l": 40, "r": 40, "t": 70, "b": 10})
        st.plotly_chart(gauge, width="stretch")
        st.caption(
            "Le pourcentage est un score de classification sur ce benchmark ; il ne constitue pas une probabilité clinique."
        )

        importance_path = REPORTS / "feature_importance.csv"
        if importance_path.exists():
            importance = pd.read_csv(importance_path).head(12).sort_values("importance")
            importance_chart = px.bar(
                importance,
                x="importance",
                y="feature",
                orientation="h",
                title="Variables les plus utilisées par le modèle",
            )
            st.plotly_chart(importance_chart, width="stretch")

with method_tab:
    st.markdown(
        """
### Pipeline

1. Validation des 178 points de chaque fenêtre EEG et reconstruction des 500 acquisitions parentes.
2. Extraction de variables temporelles (RMS, amplitude, longueur de ligne, asymétrie, kurtosis…) et spectrales (Welch, bandes delta à gamma, entropie).
3. Split 80/20 stratifié **par acquisition**, pour garder les 23 fenêtres sœurs ensemble.
4. Comparaison en validation croisée groupée avec un Dummy, une régression logistique, Random Forest et Extra Trees.
5. Sélection par Average Precision sur le train, puis évaluation finale sur le holdout avec ROC AUC, sensibilité, spécificité et F1.

### Limites à afficher dans un portfolio

- Dataset monocanal, ancien, très nettoyé et non représentatif d’un EEG clinique continu.
- 500 acquisitions originales seulement ; les 11 500 lignes sont des fenêtres, pas des patients indépendants.
- Aucune identité patient explicite : le split par acquisition réduit la fuite, mais ne garantit pas une validation patient-indépendante.
- Les cinq classes mélangent modalités d’acquisition et états physiologiques différents.
- Une performance sur ce benchmark ne permet aucune conclusion de sûreté ou d’efficacité clinique.
        """
    )

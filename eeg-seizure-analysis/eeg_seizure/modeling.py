"""Leakage-aware model training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import model_feature_columns


def recording_level_split(
    features: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified holdout where all 23 chunks of a recording stay together."""
    group_labels = features.groupby("recording_group", as_index=False)["is_seizure"].first()
    train_groups, test_groups = train_test_split(
        group_labels["recording_group"],
        test_size=test_size,
        random_state=random_state,
        stratify=group_labels["is_seizure"],
    )
    train_mask = features["recording_group"].isin(set(train_groups)).to_numpy()
    test_mask = features["recording_group"].isin(set(test_groups)).to_numpy()
    return train_mask, test_mask


def candidate_models(random_state: int = 42) -> Mapping[str, ClassifierMixin]:
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def train_candidates(
    features: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.20,
) -> dict[str, object]:
    """Train candidates on handcrafted features and return all holdout results."""
    feature_names = model_feature_columns(features)
    train_mask, test_mask = recording_level_split(features, test_size=test_size, random_state=random_state)
    x_train = features.loc[train_mask, feature_names]
    x_test = features.loc[test_mask, feature_names]
    y_train = features.loc[train_mask, "is_seizure"].to_numpy()
    y_test = features.loc[test_mask, "is_seizure"].to_numpy()

    train_groups = features.loc[train_mask, "recording_group"].to_numpy()
    cross_validation = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state)
    fitted: dict[str, ClassifierMixin] = {}
    metrics_rows: list[dict[str, float | str]] = []
    probabilities: dict[str, np.ndarray] = {}
    for name, model in candidate_models(random_state).items():
        validation_probability = cross_val_predict(
            model,
            x_train,
            y_train,
            groups=train_groups,
            cv=cross_validation,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        validation_metrics = binary_metrics(y_train, validation_probability)
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        fitted[name] = model
        probabilities[name] = probability
        metrics_rows.append(
            {
                "model": name,
                "cv_average_precision": validation_metrics["average_precision"],
                "cv_roc_auc": validation_metrics["roc_auc"],
                **binary_metrics(y_test, probability),
            }
        )

    metrics = pd.DataFrame(metrics_rows).sort_values("cv_average_precision", ascending=False).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["model"])
    best_probability = probabilities[best_name]
    predictions = features.loc[test_mask, ["segment_id", "recording_group", "y", "is_seizure", "class_name"]].copy()
    predictions["seizure_probability"] = best_probability
    predictions["prediction"] = (best_probability >= 0.5).astype(int)

    return {
        "model": fitted[best_name],
        "best_model_name": best_name,
        "metrics": metrics,
        "predictions": predictions,
        "feature_names": feature_names,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "test_probabilities": probabilities,
    }


def feature_importance(model: ClassifierMixin, feature_names: list[str]) -> pd.DataFrame:
    """Return tree importances or logistic absolute coefficients."""
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_)[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

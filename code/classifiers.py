"""Classifier factory and a SMOTE+model pipeline used across configurations A/B/C."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore


def make_classifier(name: str, seed: int = config.PRIMARY_SEED, scale_pos_weight: float | None = None):
    """Return an unfitted estimator with hyper-parameters set per Table 3.3."""
    name = name.lower()
    if name in {"lr", "logreg", "logistic_regression"}:
        return LogisticRegression(C=1.0, penalty="l2", max_iter=1000, n_jobs=-1, random_state=seed)
    if name in {"rf", "random_forest"}:
        return RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_split=4,
            n_jobs=-1, random_state=seed,
        )
    if name in {"xgb", "xgboost"}:
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            tree_method="hist", random_state=seed,
            scale_pos_weight=scale_pos_weight or 1.0,
        )
    if name in {"svm", "svc"}:
        return SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)
    if name in {"ann", "mlp"}:
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            batch_size=64,
            max_iter=100,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.15,
            random_state=seed,
        )
    raise ValueError(f"Unknown classifier '{name}'.")


CLASSIFIERS: Tuple[str, ...] = ("lr", "rf", "xgb", "svm", "ann")


def make_pipeline(name: str, seed: int = config.PRIMARY_SEED, with_smote: bool = True):
    """SMOTE-aware pipeline. Scaling is applied first, SMOTE second, model last."""
    steps = [("scaler", StandardScaler(with_mean=False))]
    if with_smote:
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("clf", make_classifier(name, seed=seed)))
    return Pipeline(steps)

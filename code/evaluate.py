"""Evaluation utilities: metrics, cross-validation, statistical tests."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from . import config

logger = logging.getLogger(__name__)

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "mcc")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict[str, float]:
    """Compute the headline metrics for a single prediction set."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_prob is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def cross_validate_metrics(
    pipe, X: np.ndarray, y: np.ndarray, n_splits: int = config.N_SPLITS_CV, seed: int = config.PRIMARY_SEED
) -> pd.DataFrame:
    """Stratified K-fold CV; return a DataFrame with one row per fold per metric."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[val_idx])
        y_prob = (
            pipe.predict_proba(X[val_idx])[:, 1]
            if hasattr(pipe.named_steps.get("clf", pipe), "predict_proba")
            else None
        )
        m = evaluate_predictions(y[val_idx], y_pred, y_prob)
        m["fold"] = fold
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical-significance tests
# ---------------------------------------------------------------------------
def paired_t_test(scores_a: Iterable[float], scores_b: Iterable[float]) -> Tuple[float, float]:
    """Paired two-sided t-test; return (t, p)."""
    a = np.asarray(list(scores_a))
    b = np.asarray(list(scores_b))
    res = stats.ttest_rel(a, b)
    return float(res.statistic), float(res.pvalue)


def wilcoxon_test(scores_a: Iterable[float], scores_b: Iterable[float]) -> Tuple[float, float]:
    a = np.asarray(list(scores_a))
    b = np.asarray(list(scores_b))
    if np.allclose(a, b):
        return 0.0, 1.0
    res = stats.wilcoxon(a, b)
    return float(res.statistic), float(res.pvalue)


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Tuple[float, float]:
    """McNemar's test on the contingency table of disagreements between two classifiers."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b_only = int(np.sum(correct_a & ~correct_b))
    a_only = int(np.sum(~correct_a & correct_b))
    n = b_only + a_only
    if n == 0:
        return 0.0, 1.0
    chi2 = ((abs(b_only - a_only) - 1) ** 2) / n
    p = 1.0 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p)


def benjamini_hochberg(pvals: List[float], fdr: float = 0.05) -> List[bool]:
    """Return a list of accept/reject flags after BH correction."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, n + 1) / n) * fdr
    passed = ranked <= thresh
    if passed.any():
        max_i = np.max(np.where(passed))
        keep = np.zeros(n, dtype=bool)
        keep[order[: max_i + 1]] = True
        return keep.tolist()
    return [False] * n

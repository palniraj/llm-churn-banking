"""SHAP explainability helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def explain_tree(model, X: np.ndarray, feature_names: Sequence[str]) -> "shap.Explanation":
    import shap

    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    if sv.values.ndim == 3:
        # binary classification: take the positive class
        sv = shap.Explanation(
            values=sv.values[..., 1],
            base_values=sv.base_values[..., 1] if sv.base_values.ndim == 2 else sv.base_values,
            data=X,
            feature_names=list(feature_names),
        )
    else:
        sv = shap.Explanation(values=sv.values, base_values=sv.base_values, data=X, feature_names=list(feature_names))
    return sv


def top_features_table(sv, top_k: int = 10) -> pd.DataFrame:
    """Return a DataFrame of mean |SHAP| value per feature, sorted descending."""
    importance = np.abs(sv.values).mean(axis=0)
    df = pd.DataFrame({"feature": list(sv.feature_names), "mean_abs_shap": importance})
    return df.sort_values("mean_abs_shap", ascending=False).head(top_k).reset_index(drop=True)


def save_summary_plot(sv, out: Path, max_display: int = 15) -> Path:
    import shap

    plt.figure()
    shap.plots.beeswarm(sv, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot -> %s", out)
    return out


def save_dependence_plot(sv, feature: str, out: Path) -> Path:
    import shap

    plt.figure()
    shap.plots.scatter(sv[:, feature], color=sv, show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("SHAP dependence plot for %s -> %s", feature, out)
    return out

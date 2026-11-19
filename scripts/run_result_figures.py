"""Produce measured result figures 4.4 (ROC) and 4.5 (confusion matrix) from the
saved models. CPU-only.

Output:
    figures/fig_4_4_roc_curves.png
    figures/fig_4_5_confusion_matrix.png
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, confusion_matrix, roc_curve, auc

from code import config

BLUE, GREEN, RED, GREY = "#1f77b4", "#2ca02c", "#d62728", "#7f7f7f"

# --- Hybrid model (measured) ---
hyb = joblib.load(config.PROCESSED_DIR / "best_hybrid_xgb.joblib")
Xte_h, yte_h = hyb["X_test"], hyb["y_test"]
prob_h = hyb["pipe"].predict_proba(Xte_h)[:, 1]
pred_h = hyb["pipe"].predict(Xte_h)

# --- Traditional model (measured) ---
trad = joblib.load(config.PROCESSED_DIR / "best_traditional.joblib")
Xte_t, yte_t = trad["X_test"], trad["y_test"]
prob_t = trad["pipe"].predict_proba(Xte_t)[:, 1]

# ---- Figure 4.4: ROC curves (Traditional vs Hybrid) ----
plt.figure(figsize=(6, 5))
for prob, yte, label, color, ls in [
    (prob_t, yte_t, "A: Traditional (RF)", BLUE, "--"),
    (prob_h, yte_h, "C: Hybrid (XGBoost)", RED, "-"),
]:
    fpr, tpr, _ = roc_curve(yte, prob)
    plt.plot(fpr, tpr, color=color, ls=ls, lw=2, label=f"{label}, AUC={auc(fpr, tpr):.3f}")
plt.plot([0, 1], [0, 1], "k:", lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves: Traditional versus Hybrid (MEASURED)", weight="bold")
plt.legend(loc="lower right", fontsize=9); plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "fig_4_4_roc_curves.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig_4_4_roc_curves.png")

# ---- Figure 4.5: Confusion matrix of Hybrid ----
cm = confusion_matrix(yte_h, pred_h)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.xticks([0, 1], ["Pred Retained", "Pred Churn"])
plt.yticks([0, 1], ["Actual Retained", "Actual Churn"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                 fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.title("Confusion Matrix: Hybrid XGBoost (MEASURED)", weight="bold")
plt.colorbar(fraction=0.045)
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / "fig_4_5_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig_4_5_confusion_matrix.png")
print("Confusion matrix:\n", cm)

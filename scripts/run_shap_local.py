"""Generate the real SHAP summary figure + top-feature table from the saved
Hybrid XGBoost model. CPU-only, no GPU needed.

Output:
    data/processed/shap_top_hybrid.csv
    figures/fig_4_6_shap_summary.png
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
import shap

from code import config

art = joblib.load(config.PROCESSED_DIR / "best_hybrid_xgb.joblib")
pipe, names, X_test = art["pipe"], art["feature_names"], art["X_test"]
print(f"Model loaded. Test set: {X_test.shape}, {len(names)} features.")

# Apply only the scaler (SMOTE is a train-time resampler with no transform at predict)
X_pre = pipe.named_steps["scaler"].transform(X_test)
explainer = shap.TreeExplainer(pipe.named_steps["clf"])
sv = explainer(X_pre)

# Global importance
importance = np.abs(sv.values).mean(axis=0)
order = np.argsort(importance)[::-1]
top = [(names[i], float(importance[i])) for i in order[:15]]

import pandas as pd
pd.DataFrame(top, columns=["feature", "mean_abs_shap"]).to_csv(
    config.PROCESSED_DIR / "shap_top_hybrid.csv", index=False
)
print("\nTop-15 features by mean |SHAP|:")
for f, v in top:
    print(f"  {f:>22s}  {v:.4f}")

# Readable summary bar plot (beeswarm can be heavy; use a clean bar chart)
feats = [t[0] for t in top][::-1]
vals = [t[1] for t in top][::-1]
plt.figure(figsize=(7.5, 6))
plt.barh(range(len(feats)), vals, color="#1f77b4", alpha=0.85)
plt.yticks(range(len(feats)), feats, fontsize=8)
plt.xlabel("Mean |SHAP value|")
plt.title("SHAP Summary: Hybrid XGBoost (Top-15 Features, MEASURED)", weight="bold")
plt.tight_layout()
out = config.FIGURES_DIR / "fig_4_6_shap_summary.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")

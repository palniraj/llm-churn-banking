"""One-command defense demo. Reproduces the headline MEASURED results in ~1 minute
on a CPU-only laptop, and prints the saved cross-configuration comparison and the
SHAP top-feature ranking.

Usage (from the thesis/ directory):
    .venv/bin/python scripts/demo.py
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from code import classifiers, config, data_loader, evaluate, preprocess

BAR = "=" * 68


def section(title):
    print("\n" + BAR)
    print(title)
    print(BAR)


def main():
    section("1. DATA — Bank Customer Churn (primary dataset)")
    bank = preprocess.clean_bank_churn(data_loader.load_bank_churn())
    print(f"Records: {len(bank):,}   Features: {bank.shape[1]-2}   "
          f"Churn rate: {bank[config.TARGET_COL].mean():.3f}")

    section("2. LIVE — Configuration A (traditional), Random Forest, 5-fold CV")
    X, names = preprocess.encode_traditional(bank.drop(columns=[config.TARGET_COL]))
    y = bank[config.TARGET_COL].astype(int).values
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.PRIMARY_SEED)
    rows = []
    for tr, va in skf.split(X.values, y):
        pipe = classifiers.make_pipeline("rf")
        pipe.fit(X.values[tr], y[tr])
        yp = pipe.predict(X.values[va]); pp = pipe.predict_proba(X.values[va])[:, 1]
        rows.append(evaluate.evaluate_predictions(y[va], yp, pp))
    cv = pd.DataFrame(rows).mean()
    print(f"Random Forest  ->  Accuracy {cv['accuracy']:.3f} | F1 {cv['f1']:.3f} | "
          f"ROC-AUC {cv['roc_auc']:.3f} | MCC {cv['mcc']:.3f}")
    print("(Table 9 reports 0.608 F1 from 10-fold x 3 seeds; this is 5-fold, single seed)")

    section("3. SAVED — Cross-configuration comparison (Table 12)")
    t = config.PROCESSED_DIR / "table_4_6.csv"
    if t.exists():
        print(pd.read_csv(t).to_string(index=False))
    else:
        print("table_4_6.csv not found — run scripts/run_hybrid_light.py first.")

    section("4. SAVED — SHAP top features, Hybrid model (Table 13)")
    s = config.PROCESSED_DIR / "shap_top_hybrid.csv"
    if s.exists():
        print(pd.read_csv(s).head(10).to_string(index=False))
        print("\nKey finding: 'avg_sentiment' (CFPB complaint feature) is the #1 driver.")
    else:
        print("shap_top_hybrid.csv not found — run scripts/run_shap_local.py first.")

    section("5. SAVED — Out-of-distribution validation (Section 4.3.6)")
    o = config.PROCESSED_DIR / "ood_metrics.csv"
    if o.exists():
        print(pd.read_csv(o).to_string(index=False))

    print("\n" + BAR)
    print("DEMO COMPLETE. Figures are in figures/. Full results in data/processed/.")
    print(BAR)


if __name__ == "__main__":
    main()

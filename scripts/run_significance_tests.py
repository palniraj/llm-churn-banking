"""Paired statistical comparison of Configuration A against Configuration C.

Section 4.3.4 of the report claims a significance verdict, and Table 11 quotes a
mean +/- standard deviation for the Hybrid model. Neither could previously be
reproduced: `table_4_5_light.csv` stores only fold means (no spread), and the
only saved test output, `table_4_7.csv`, was computed on an earlier
Configuration C than the one now reported.

This script fixes both in a single seeded run. Configuration A (Random Forest,
the best model of Table 9) and Configuration C (XGBoost, Table 11) are
cross-validated over *identical* folds, so their per-fold scores are genuinely
paired, then compared with the tests specified in Section 3.5.2.

Fold count and XGBoost hyper-parameters match `run_hybrid_light.py`, so
Configuration C here is the same estimator that produced
`best_hybrid_xgb.joblib`.

Output:
    data/processed/cv_folds_A_vs_C.csv      per-fold metrics for both configs
    data/processed/table_4_7_paired.csv     t-test / Wilcoxon / McNemar + BH
"""
from __future__ import annotations

import os
# Cap threads BEFORE numpy/xgboost import so the laptop stays responsive.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from code import classifiers, config, data_loader, evaluate, fusion, preprocess

N_SPLITS = 5          # matches run_hybrid_light.py
SEED = config.PRIMARY_SEED


def build_matrices():
    """Configuration A (traditional only) and C (traditional + complaint + ST)."""
    merged = pd.read_parquet(config.PROCESSED_DIR / "bank_merged.parquet")
    y_c = merged[config.TARGET_COL].astype(int).values

    drop = [config.TARGET_COL, "product_bucket", "region"]
    df_X = merged.drop(columns=[c for c in drop if c in merged.columns])

    # --- Configuration C: traditional + complaint features + ST embeddings ---
    X_trad_c, _ = preprocess.encode_traditional(df_X)
    for c in fusion.get_complaint_feature_columns():
        if c not in X_trad_c.columns and c in df_X.columns:
            X_trad_c[c] = df_X[c].astype(float).values
    X_emb = np.load(config.PROCESSED_DIR / "bank_emb_st.npy").astype(float)
    X_C = np.concatenate([X_trad_c.values, X_emb], axis=1)

    # --- Configuration A: traditional only, straight from the raw CSV ---
    bank = preprocess.clean_bank_churn(data_loader.load_bank_churn())
    X_A_df, _ = preprocess.encode_traditional(bank.drop(columns=[config.TARGET_COL]))
    X_A = X_A_df.values
    y_a = bank[config.TARGET_COL].astype(int).values

    # The paired test is only valid if both matrices describe the same customers
    # in the same order.
    if not np.array_equal(y_a, y_c):
        raise SystemExit(
            "Row order differs between bank_churn.csv and bank_merged.parquet; "
            "the folds would not be paired. Aborting rather than reporting an "
            "invalid test."
        )
    return X_A, X_C, y_c


def hybrid_pipe():
    """Identical to run_hybrid_light.py so Table 11 stays consistent."""
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("smote", SMOTE(random_state=SEED)),
        ("clf", XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            tree_method="hist", n_jobs=2, random_state=SEED,
        )),
    ])


def paired_cv(X_A, X_C, y):
    """Cross-validate both configurations over the same folds."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, va) in enumerate(skf.split(X_A, y)):
        for cfg, X, pipe in (("A", X_A, classifiers.make_pipeline("rf")),
                             ("C", X_C, hybrid_pipe())):
            pipe.fit(X[tr], y[tr])
            pred = pipe.predict(X[va])
            prob = pipe.predict_proba(X[va])[:, 1]
            m = evaluate.evaluate_predictions(y[va], pred, prob)
            m.update(config=cfg, fold=fold)
            rows.append(m)
        print(f"  fold {fold}: "
              f"A f1={rows[-2]['f1']:.4f}  C f1={rows[-1]['f1']:.4f}")
    return pd.DataFrame(rows)


def main():
    X_A, X_C, y = build_matrices()
    print(f"Configuration A: {X_A.shape}   Configuration C: {X_C.shape}   "
          f"churn rate {y.mean():.4f}")
    print(f"\n{N_SPLITS}-fold paired cross-validation (seed {SEED}, 2 threads):")
    cv = paired_cv(X_A, X_C, y)
    cv.to_csv(config.PROCESSED_DIR / "cv_folds_A_vs_C.csv", index=False)

    print("\nPer-configuration summary (mean +/- sample std over folds):")
    for cfg in ("A", "C"):
        sub = cv[cv.config == cfg]
        parts = [f"{k}={sub[k].mean():.3f}+/-{sub[k].std(ddof=1):.3f}"
                 for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "mcc")]
        print(f"  {cfg}: " + "  ".join(parts))

    # ---- paired tests on per-fold F1 -----------------------------------
    f1_a = cv[cv.config == "A"].sort_values("fold")["f1"].values
    f1_c = cv[cv.config == "C"].sort_values("fold")["f1"].values
    t, p_t = evaluate.paired_t_test(f1_c, f1_a)
    w, p_w = evaluate.wilcoxon_test(f1_c, f1_a)

    # ---- McNemar on a common held-out split -----------------------------
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=config.TEST_SIZE, stratify=y,
                                random_state=SEED)
    pa = classifiers.make_pipeline("rf").fit(X_A[itr], y[itr]).predict(X_A[ite])
    pc = hybrid_pipe().fit(X_C[itr], y[itr]).predict(X_C[ite])
    chi2, p_mc = evaluate.mcnemar_test(y[ite], pa, pc)

    flags = evaluate.benjamini_hochberg([p_t, p_w, p_mc], fdr=0.05)
    out = pd.DataFrame([
        {"test": "paired t-test (per-fold F1)", "statistic": round(t, 4),
         "p_value": round(p_t, 6), "bh_significant": bool(flags[0])},
        {"test": "Wilcoxon signed-rank (per-fold F1)", "statistic": round(w, 4),
         "p_value": round(p_w, 6), "bh_significant": bool(flags[1])},
        {"test": "McNemar (held-out predictions)", "statistic": round(chi2, 4),
         "p_value": round(p_mc, 6), "bh_significant": bool(flags[2])},
    ])
    out.to_csv(config.PROCESSED_DIR / "table_4_7_paired.csv", index=False)

    print("\nHybrid (C) versus Traditional (A), Benjamini-Hochberg at FDR 0.05:")
    print(out.to_string(index=False))
    print(f"\nmean F1  A={f1_a.mean():.4f}  C={f1_c.mean():.4f}  "
          f"difference={f1_c.mean() - f1_a.mean():+.4f}")
    print("\nWrote cv_folds_A_vs_C.csv and table_4_7_paired.csv")


if __name__ == "__main__":
    main()

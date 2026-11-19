"""CPU-safe, single-question run: does the FULL Hybrid feature set
(traditional + complaint-derived + Sentence-Transformer embeddings) beat the
measured traditional baseline?

Kept deliberately light so it does not lock a laptop:
  - XGBoost only (no SVM, no ANN)
  - 5-fold CV (not 10)
  - capped to 2 threads

Output:
    data/processed/table_4_5_light.csv
    data/processed/best_hybrid_xgb.joblib
    data/processed/table_4_6.csv
"""
from __future__ import annotations

import os
# Cap threads BEFORE importing numpy/xgboost so the laptop stays responsive.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from code import config, evaluate, fusion, preprocess


def build_hybrid():
    merged = pd.read_parquet(config.PROCESSED_DIR / "bank_merged.parquet")
    y = merged[config.TARGET_COL].astype(int).values

    drop = [config.TARGET_COL, "product_bucket", "region"]
    df_X = merged.drop(columns=[c for c in drop if c in merged.columns])
    X_trad, names_trad = preprocess.encode_traditional(df_X)
    for c in fusion.get_complaint_feature_columns():
        if c not in X_trad.columns and c in df_X.columns:
            X_trad[c] = df_X[c].astype(float).values
    names_trad = list(X_trad.columns)

    X_emb = np.load(config.PROCESSED_DIR / "bank_emb_st.npy").astype(float)
    names_emb = [f"st_emb_{i}" for i in range(X_emb.shape[1])]

    X_C = np.concatenate([X_trad.values, X_emb], axis=1)
    return X_C, names_trad + names_emb, y


def xgb_pipe():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("smote", SMOTE(random_state=config.PRIMARY_SEED)),
        ("clf", XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            tree_method="hist", n_jobs=2, random_state=config.PRIMARY_SEED,
        )),
    ])


def cv5(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.PRIMARY_SEED)
    rows = []
    for tr, va in skf.split(X, y):
        pipe = xgb_pipe()
        pipe.fit(X[tr], y[tr])
        yp = pipe.predict(X[va])
        pp = pipe.predict_proba(X[va])[:, 1]
        rows.append(evaluate.evaluate_predictions(y[va], yp, pp))
    return pd.DataFrame(rows)


def main():
    X_C, names_C, y = build_hybrid()
    print(f"Hybrid feature matrix: {X_C.shape}  (churn {y.mean():.4f})")
    print("Running 5-fold CV with XGBoost (capped to 2 threads)...")
    cv = cv5(X_C, y)
    summary = {k: (round(cv[k].mean(), 3), round(cv[k].std(ddof=1), 3)) for k in cv.columns}
    print("\nHybrid XGBoost (MEASURED, 5-fold):")
    for k, (m, s) in summary.items():
        print(f"  {k:>10s}: {m:.3f} +/- {s:.3f}")

    pd.DataFrame([{**{k: cv[k].mean() for k in cv.columns}}]).to_csv(
        config.PROCESSED_DIR / "table_4_5_light.csv", index=False
    )

    # Compare against measured A and B best-F1
    def best_f1(path):
        d = pd.read_csv(config.PROCESSED_DIR / path)
        r = d[d.metric == "f1"].sort_values("mean", ascending=False).head(1)
        clf = r.clf.iloc[0]
        means = d[d.clf == clf].set_index("metric")["mean"]
        return clf, means

    a_clf, a = best_f1("cv_results_A.csv")
    b_clf, b = best_f1("cv_results_B.csv")
    rows_46 = [
        {"config": "A (traditional)", "best_clf": a_clf, "accuracy": round(a["accuracy"], 3),
         "f1": round(a["f1"], 3), "roc_auc": round(a["roc_auc"], 3), "mcc": round(a["mcc"], 3)},
        {"config": "B (LLM-only)", "best_clf": b_clf, "accuracy": round(b["accuracy"], 3),
         "f1": round(b["f1"], 3), "roc_auc": round(b["roc_auc"], 3), "mcc": round(b["mcc"], 3)},
        {"config": "C (hybrid+complaints)", "best_clf": "xgb",
         "accuracy": round(cv["accuracy"].mean(), 3), "f1": round(cv["f1"].mean(), 3),
         "roc_auc": round(cv["roc_auc"].mean(), 3), "mcc": round(cv["mcc"].mean(), 3)},
    ]
    table46 = pd.DataFrame(rows_46)
    table46.to_csv(config.PROCESSED_DIR / "table_4_6.csv", index=False)
    print("\nTable 4.6 (cross-config best, MEASURED):")
    print(table46.to_string(index=False))

    # Persist best hybrid model
    pipe = xgb_pipe()
    Xtr, Xte, ytr, yte = train_test_split(
        X_C, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.PRIMARY_SEED
    )
    pipe.fit(Xtr, ytr)
    joblib.dump({"pipe": pipe, "feature_names": names_C, "X_test": Xte, "y_test": yte},
                config.PROCESSED_DIR / "best_hybrid_xgb.joblib")
    print("\nSaved best hybrid pipeline.")


if __name__ == "__main__":
    main()

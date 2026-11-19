"""Run Configuration C (Hybrid: traditional + Sentence-Transformer embeddings)
without the slow SVM step (it is computed separately if time permits).

Output:
    data/processed/cv_results_C.csv
    data/processed/table_4_5.csv
    data/processed/best_hybrid_xgb.joblib
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from code import classifiers, config, data_loader, evaluate, preprocess


def _summary_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    pm = df.pivot_table(index="clf", columns="metric", values="mean")
    ps = df.pivot_table(index="clf", columns="metric", values="std")
    out = pd.DataFrame(index=pm.index)
    for col in ("accuracy", "precision", "recall", "f1", "roc_auc", "mcc"):
        if col in pm.columns:
            out[col] = pm[col].round(3).astype(str) + " +/- " + ps[col].round(3).astype(str)
    return out


def main():
    bank = preprocess.clean_bank_churn(data_loader.load_bank_churn())
    y = bank[config.TARGET_COL].astype(int).values

    X_A, names_A = preprocess.encode_traditional(bank.drop(columns=[config.TARGET_COL]))
    X_A = X_A.values
    X_B = np.load(config.PROCESSED_DIR / "bank_emb_st.npy").astype(float)
    names_B = [f"st_emb_{i}" for i in range(X_B.shape[1])]
    X_C = np.concatenate([X_A, X_B], axis=1)
    names_C = list(names_A) + names_B
    print(f"Shapes  A:{X_A.shape}  B:{X_B.shape}  C:{X_C.shape}")

    rows = []
    # Skip SVM (slow on 395-d w/ 10-fold). LR, RF, XGB, ANN cover the headline points.
    fast_clfs = ("lr", "rf", "xgb", "ann")
    for clf in fast_clfs:
        pipe = classifiers.make_pipeline(clf)
        cv = evaluate.cross_validate_metrics(pipe, X_C, y)
        for metric in evaluate.METRICS:
            if metric in cv.columns:
                rows.append({
                    "config": "C", "clf": clf, "metric": metric,
                    "mean": float(cv[metric].mean()),
                    "std": float(cv[metric].std(ddof=1)),
                })
        print(f"  {clf:>4s}  acc={cv['accuracy'].mean():.3f}  f1={cv['f1'].mean():.3f}  auc={cv['roc_auc'].mean():.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(config.PROCESSED_DIR / "cv_results_C.csv", index=False)
    table = _summary_table(rows)
    table.to_csv(config.PROCESSED_DIR / "table_4_5.csv")
    print()
    print(table)

    # Persist best Hybrid model
    best_clf = df[df.metric == "f1"].sort_values("mean", ascending=False).head(1).clf.iloc[0]
    print(f"\nBest Hybrid classifier: {best_clf}")
    pipe = classifiers.make_pipeline(best_clf)
    X_train, X_test, y_train, y_test = train_test_split(
        X_C, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.PRIMARY_SEED
    )
    pipe.fit(X_train, y_train)
    out = config.PROCESSED_DIR / "best_hybrid_xgb.joblib"
    joblib.dump({"pipe": pipe, "feature_names": names_C, "X_test": X_test, "y_test": y_test}, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

"""Run Configuration A (traditional features only) on the bank-churn dataset.

This is the CPU-only / no-LLM baseline. It produces:
    data/processed/cv_results_A.csv     # per-fold metrics for every classifier
    data/processed/table_4_3.csv        # summary table mirroring Table 4.3
    data/processed/best_traditional.joblib   # the best Configuration A pipeline
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


def main():
    bank = preprocess.clean_bank_churn(data_loader.load_bank_churn())
    X_df, names = preprocess.encode_traditional(bank.drop(columns=[config.TARGET_COL]))
    X = X_df.values
    y = bank[config.TARGET_COL].astype(int).values
    print(f"Features: {X.shape}, churn rate: {y.mean():.4f}")

    rows = []
    fold_f1 = {}
    for clf_name in classifiers.CLASSIFIERS:
        pipe = classifiers.make_pipeline(clf_name)
        cv = evaluate.cross_validate_metrics(pipe, X, y)
        fold_f1[clf_name] = cv["f1"].values
        for metric in evaluate.METRICS:
            if metric in cv.columns:
                rows.append({
                    "config": "A",
                    "clf": clf_name,
                    "metric": metric,
                    "mean": float(cv[metric].mean()),
                    "std": float(cv[metric].std(ddof=1)),
                })
        print(f"  {clf_name:>4s}  acc={cv['accuracy'].mean():.3f}  f1={cv['f1'].mean():.3f}  auc={cv['roc_auc'].mean():.3f}")

    results = pd.DataFrame(rows)
    cv_path = config.PROCESSED_DIR / "cv_results_A.csv"
    results.to_csv(cv_path, index=False)
    print(f"\nWrote {cv_path}")

    # Summary table mirroring Table 4.3 (mean +/- std)
    pivot_mean = results.pivot_table(index="clf", columns="metric", values="mean")
    pivot_std = results.pivot_table(index="clf", columns="metric", values="std")
    table = pd.DataFrame(index=pivot_mean.index)
    for col in ("accuracy", "precision", "recall", "f1", "roc_auc", "mcc"):
        if col in pivot_mean.columns:
            table[col] = pivot_mean[col].round(3).astype(str) + " +/- " + pivot_std[col].round(3).astype(str)
    table_path = config.PROCESSED_DIR / "table_4_3.csv"
    table.to_csv(table_path)
    print(f"Wrote {table_path}")
    print(table)

    # Persist the best Configuration A model on a final 70/15/15 hold-out
    best_clf = max(fold_f1, key=lambda k: fold_f1[k].mean())
    print(f"\nBest classifier on Configuration A: {best_clf}")
    pipe = classifiers.make_pipeline(best_clf)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.PRIMARY_SEED
    )
    pipe.fit(X_train, y_train)
    out = config.PROCESSED_DIR / "best_traditional.joblib"
    joblib.dump({"pipe": pipe, "feature_names": names, "X_test": X_test, "y_test": y_test}, out)
    print(f"Saved best Configuration A model -> {out}")


if __name__ == "__main__":
    main()

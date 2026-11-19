"""Out-of-distribution validation: fit Configuration A on the primary bank-churn
dataset and evaluate (without any further training) on the S4E1 Bank Churn dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from code import classifiers, config, data_loader, evaluate, preprocess


def _align_features(target_cols: list[str], df: pd.DataFrame) -> np.ndarray:
    """Align an OOD frame's columns to the training frame's column order; missing -> 0."""
    out = pd.DataFrame(0.0, index=df.index, columns=target_cols)
    for col in target_cols:
        if col in df.columns:
            out[col] = df[col]
    return out.astype(float).values


def main():
    bank = preprocess.clean_bank_churn(data_loader.load_bank_churn())
    X_train_df, train_cols = preprocess.encode_traditional(bank.drop(columns=[config.TARGET_COL]))
    y_train = bank[config.TARGET_COL].astype(int).values

    # fit Random Forest (best on Configuration A) on the entire primary set
    pipe = classifiers.make_pipeline("rf")
    pipe.fit(X_train_df.values, y_train)

    # Load OOD set (S4E1)
    ood = data_loader.load_maven_churn()
    ood = preprocess.clean_bank_churn(ood)
    print("OOD shape:", ood.shape, "churn rate:", ood[config.TARGET_COL].mean().round(4))

    # Encode using the same one-hot scheme as training (drop_first=True)
    X_ood_df, ood_cols = preprocess.encode_traditional(ood.drop(columns=[config.TARGET_COL]))
    y_ood = ood[config.TARGET_COL].astype(int).values

    # Align columns to training order
    X_ood = _align_features(train_cols, X_ood_df)
    print(f"Train cols: {len(train_cols)}; OOD cols (encoded): {len(ood_cols)}; aligned: {X_ood.shape[1]}")

    y_pred = pipe.predict(X_ood)
    y_prob = pipe.predict_proba(X_ood)[:, 1]
    metrics = evaluate.evaluate_predictions(y_ood, y_pred, y_prob)

    print("\nOOD performance (RF trained on bank_churn, evaluated on S4E1):")
    for k, v in metrics.items():
        print(f"  {k:>10s}: {v:.4f}")

    out = config.PROCESSED_DIR / "ood_metrics.csv"
    pd.DataFrame([metrics]).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

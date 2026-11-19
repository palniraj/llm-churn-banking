"""Run Configuration C with the FULL Hybrid feature set:
traditional features + Sentence-Transformer embeddings + complaint-derived features.

Output:
    data/processed/cv_results_C_full.csv
    data/processed/table_4_5.csv         (overwrites)
    data/processed/best_hybrid_xgb.joblib (overwrites)
    data/processed/table_4_6.csv         (cross-config best models)
    data/processed/table_4_7.csv         (paired t-tests)
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

from code import classifiers, config, evaluate, fusion, preprocess


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
    merged = pd.read_parquet(config.PROCESSED_DIR / "bank_merged.parquet")
    y = merged[config.TARGET_COL].astype(int).values
    print(f"Merged shape: {merged.shape}")

    # Drop columns that are not features (target + helper keys)
    drop = [config.TARGET_COL, "product_bucket", "region"]
    df_X = merged.drop(columns=[c for c in drop if c in merged.columns])
    X_traditional, names_trad = preprocess.encode_traditional(df_X)
    # encode_traditional drops age_band; keep complaint cols
    complaint_cols = fusion.get_complaint_feature_columns()
    for c in complaint_cols:
        if c not in X_traditional.columns and c in df_X.columns:
            X_traditional[c] = df_X[c].astype(float).values
    X_traditional_arr = X_traditional.values
    names_trad = list(X_traditional.columns)
    print(f"Traditional + complaint features: {X_traditional_arr.shape[1]}  "
          f"(names sample: {names_trad[:8]} ... {names_trad[-4:]})")

    X_emb = np.load(config.PROCESSED_DIR / "bank_emb_st.npy").astype(float)
    names_emb = [f"st_emb_{i}" for i in range(X_emb.shape[1])]
    X_C = np.concatenate([X_traditional_arr, X_emb], axis=1)
    names_C = names_trad + names_emb
    print(f"Hybrid (full): {X_C.shape}")

    rows = []
    fold_f1_C = None
    for clf in ("lr", "rf", "xgb", "ann"):
        pipe = classifiers.make_pipeline(clf)
        cv = evaluate.cross_validate_metrics(pipe, X_C, y)
        if clf == "xgb":
            fold_f1_C = cv["f1"].values
        for metric in evaluate.METRICS:
            if metric in cv.columns:
                rows.append({
                    "config": "C", "clf": clf, "metric": metric,
                    "mean": float(cv[metric].mean()),
                    "std": float(cv[metric].std(ddof=1)),
                })
        print(f"  {clf:>4s}  acc={cv['accuracy'].mean():.3f}  f1={cv['f1'].mean():.3f}  auc={cv['roc_auc'].mean():.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(config.PROCESSED_DIR / "cv_results_C_full.csv", index=False)
    table = _summary_table(rows)
    table.to_csv(config.PROCESSED_DIR / "table_4_5.csv")
    print("\nTable 4.5 (Hybrid):")
    print(table)

    # Table 4.6 – cross-config best models
    rows_46 = []
    for cfg, path in (("A", "cv_results_A.csv"),
                      ("B", "cv_results_B.csv"),
                      ("C", "cv_results_C_full.csv")):
        d = pd.read_csv(config.PROCESSED_DIR / path)
        f1_d = d[d.metric == "f1"].sort_values("mean", ascending=False).head(1)
        clf = f1_d.clf.iloc[0]
        means = d[d.clf == clf].set_index("metric")["mean"]
        rows_46.append({
            "config": cfg,
            "best_clf": clf,
            "accuracy": round(means.get("accuracy", float("nan")), 3),
            "f1": round(means["f1"], 3),
            "roc_auc": round(means.get("roc_auc", float("nan")), 3),
            "mcc": round(means.get("mcc", float("nan")), 3),
        })
    table46 = pd.DataFrame(rows_46)
    table46.to_csv(config.PROCESSED_DIR / "table_4_6.csv", index=False)
    print("\nTable 4.6 (cross-config best):")
    print(table46.to_string(index=False))

    # Table 4.7 – paired t-tests on best-config-per-cfg per-fold F1
    bank = preprocess.clean_bank_churn(__import__("code").data_loader.load_bank_churn())
    y2 = bank[config.TARGET_COL].astype(int).values
    X_A_only, _ = preprocess.encode_traditional(bank.drop(columns=[config.TARGET_COL]))
    X_A_only = X_A_only.values

    def best_clf_f1(X, results_path):
        d = pd.read_csv(config.PROCESSED_DIR / results_path)
        clf = d[d.metric == "f1"].sort_values("mean", ascending=False).head(1).clf.iloc[0]
        cv = evaluate.cross_validate_metrics(classifiers.make_pipeline(clf), X, y)
        return clf, cv["f1"].values

    a_clf, a_f1 = best_clf_f1(X_A_only, "cv_results_A.csv")
    b_clf, b_f1 = best_clf_f1(X_emb,    "cv_results_B.csv")
    c_clf, c_f1 = best_clf_f1(X_C,      "cv_results_C_full.csv")
    print(f"\nFold F1 means: A({a_clf})={a_f1.mean():.3f}  "
          f"B({b_clf})={b_f1.mean():.3f}  C({c_clf})={c_f1.mean():.3f}")

    sig_rows = []
    for label, x_arr, y_arr in [
        ("Hybrid vs Traditional",   c_f1, a_f1),
        ("Hybrid vs LLM-only",      c_f1, b_f1),
        ("LLM-only vs Traditional", b_f1, a_f1),
    ]:
        t, p = evaluate.paired_t_test(x_arr, y_arr)
        w, pw = evaluate.wilcoxon_test(x_arr, y_arr)
        sig_rows.append({
            "comparison": label, "t": round(t, 3), "p_t": round(p, 6),
            "wilcoxon": round(w, 3), "p_wilcoxon": round(pw, 6),
        })
    sig_df = pd.DataFrame(sig_rows)
    sig_df["bh_significant"] = evaluate.benjamini_hochberg(sig_df.p_t.tolist(), fdr=0.05)
    sig_df.to_csv(config.PROCESSED_DIR / "table_4_7.csv", index=False)
    print("\nTable 4.7 (significance):")
    print(sig_df.to_string(index=False))

    # Best Hybrid model
    print(f"\nBest Hybrid classifier: {c_clf}")
    pipe = classifiers.make_pipeline(c_clf)
    X_train, X_test, y_train, y_test = train_test_split(
        X_C, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.PRIMARY_SEED
    )
    pipe.fit(X_train, y_train)
    out = config.PROCESSED_DIR / "best_hybrid_xgb.joblib"
    joblib.dump({"pipe": pipe, "feature_names": names_C, "X_test": X_test, "y_test": y_test}, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

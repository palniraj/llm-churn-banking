"""Run Configurations B (LLM-embed only) and C (Hybrid) with measured 10-fold CV.

Output:
    data/processed/cv_results_B.csv
    data/processed/cv_results_C.csv
    data/processed/table_4_4.csv
    data/processed/table_4_5.csv
    data/processed/table_4_6.csv      (cross-config comparison)
    data/processed/table_4_7.csv      (paired t-test on F1 across configs)
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

    print(f"Shapes  A:{X_A.shape}  B:{X_B.shape}  C:{X_C.shape}  y_pos={y.mean():.4f}")

    fold_f1 = {}
    for cfg, X, suffix in [("B", X_B, "B"), ("C", X_C, "C")]:
        rows = []
        print(f"\n=== Configuration {cfg} ===")
        for clf in classifiers.CLASSIFIERS:
            pipe = classifiers.make_pipeline(clf)
            cv = evaluate.cross_validate_metrics(pipe, X, y)
            fold_f1[(cfg, clf)] = cv["f1"].values
            for metric in evaluate.METRICS:
                if metric in cv.columns:
                    rows.append({
                        "config": cfg, "clf": clf, "metric": metric,
                        "mean": float(cv[metric].mean()),
                        "std": float(cv[metric].std(ddof=1)),
                    })
            print(f"  {clf:>4s}  acc={cv['accuracy'].mean():.3f}  f1={cv['f1'].mean():.3f}  auc={cv['roc_auc'].mean():.3f}")

        df = pd.DataFrame(rows)
        df.to_csv(config.PROCESSED_DIR / f"cv_results_{suffix}.csv", index=False)
        table = _summary_table(rows)
        out = config.PROCESSED_DIR / f"table_4_{4 if cfg == 'B' else 5}.csv"
        table.to_csv(out)
        print(table)

    # Cross-config Table 4.6 + paired t-tests (Table 4.7)
    A_results = pd.read_csv(config.PROCESSED_DIR / "cv_results_A.csv")
    A_f1 = A_results[A_results.metric == "f1"]
    rows_46 = []
    for cfg, all_results_path in [
        ("A", config.PROCESSED_DIR / "cv_results_A.csv"),
        ("B", config.PROCESSED_DIR / "cv_results_B.csv"),
        ("C", config.PROCESSED_DIR / "cv_results_C.csv"),
    ]:
        d = pd.read_csv(all_results_path)
        f1_only = d[d.metric == "f1"].sort_values("mean", ascending=False).head(1)
        clf = f1_only.clf.iloc[0]
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
    print(table46)

    # Paired t-tests on per-fold F1 of each configuration's best classifier
    # Re-run CV for the best classifier of each configuration to get fold-aligned F1
    def _best_clf_f1(X, results_path):
        d = pd.read_csv(results_path)
        clf = d[d.metric == "f1"].sort_values("mean", ascending=False).head(1).clf.iloc[0]
        pipe = classifiers.make_pipeline(clf)
        cv = evaluate.cross_validate_metrics(pipe, X, y)
        return clf, cv["f1"].values

    a_clf, a_f1 = _best_clf_f1(X_A, config.PROCESSED_DIR / "cv_results_A.csv")
    b_clf, b_f1 = _best_clf_f1(X_B, config.PROCESSED_DIR / "cv_results_B.csv")
    c_clf, c_f1 = _best_clf_f1(X_C, config.PROCESSED_DIR / "cv_results_C.csv")

    sig_rows = []
    for label, x, y_arr in [
        ("Hybrid vs Traditional", c_f1, a_f1),
        ("Hybrid vs LLM-only",    c_f1, b_f1),
        ("LLM-only vs Traditional", b_f1, a_f1),
    ]:
        t, p = evaluate.paired_t_test(x, y_arr)
        w, pw = evaluate.wilcoxon_test(x, y_arr)
        sig_rows.append({
            "comparison": label, "t": round(t, 3), "p_t": round(p, 6),
            "wilcoxon": round(w, 3), "p_wilcoxon": round(pw, 6),
        })
    sig_df = pd.DataFrame(sig_rows)
    sig_df["bh_significant"] = evaluate.benjamini_hochberg(sig_df.p_t.tolist(), fdr=0.05)
    sig_df.to_csv(config.PROCESSED_DIR / "table_4_7.csv", index=False)
    print("\nTable 4.7 (paired tests):")
    print(sig_df)

    # Persist the best Hybrid pipeline for SHAP & retention generation
    print(f"\nBest Hybrid classifier: {c_clf}")
    pipe = classifiers.make_pipeline(c_clf)
    X_train, X_test, y_train, y_test = train_test_split(
        X_C, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.PRIMARY_SEED
    )
    pipe.fit(X_train, y_train)
    art = {"pipe": pipe, "feature_names": names_C, "X_test": X_test, "y_test": y_test}
    out = config.PROCESSED_DIR / "best_hybrid_xgb.joblib"
    joblib.dump(art, out)
    print(f"Saved best Hybrid pipeline -> {out}")


if __name__ == "__main__":
    main()

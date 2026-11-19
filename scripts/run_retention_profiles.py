"""Stage 1 of the retention-message evaluation: build the 50 rated profiles.

Selects the at-risk customers from the Hybrid model's held-out set, extracts the
SHAP drivers behind each prediction, renders them into readable phrases, and
writes one row per profile containing the prompt for the generative model and
the generic baseline message it will be compared against.

Runs on CPU in seconds. No generative model is needed at this stage.

Output:
    data/processed/retention_profiles.csv
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from code import config, fusion, preprocess, retention_llm

N_PROFILES = 50
RISK_THRESHOLD = 0.7          # Section 3.2.4
TOP_K_DRIVERS = 5
SEED = config.PRIMARY_SEED

# The bank's stated guardrails, passed to the model with every prompt.
POLICY = (
    "Offers limited to: fee waiver review, no-fee credit card eligibility check, "
    "complimentary financial review with an adviser, and product consolidation. "
    "Do not promise specific monetary amounts, interest rates, or products the "
    "bank does not offer. Do not reference the customer's complaint history "
    "directly."
)


def rebuild_hybrid_and_split():
    """Reproduce the exact matrix and 85/15 split used to fit the saved model."""
    merged = pd.read_parquet(config.PROCESSED_DIR / "bank_merged.parquet")
    y = merged[config.TARGET_COL].astype(int).values

    drop = [config.TARGET_COL, "product_bucket", "region"]
    df_X = merged.drop(columns=[c for c in drop if c in merged.columns])
    X_trad, _ = preprocess.encode_traditional(df_X)
    for c in fusion.get_complaint_feature_columns():
        if c not in X_trad.columns and c in df_X.columns:
            X_trad[c] = df_X[c].astype(float).values
    emb = np.load(config.PROCESSED_DIR / "bank_emb_st.npy").astype(float)
    X_C = np.concatenate([X_trad.values, emb], axis=1)

    idx = np.arange(len(y))
    _, test_idx = train_test_split(
        idx, test_size=config.TEST_SIZE, stratify=y, random_state=SEED
    )
    return merged, X_C, y, test_idx


def stratified_sample(frame, n, seed=SEED):
    """Sample n profiles stratified by churn-probability band and geography."""
    frame = frame.copy()
    frame["risk_band"] = pd.cut(
        frame["churn_probability"], [0.7, 0.8, 0.9, 1.0],
        labels=["0.70-0.80", "0.80-0.90", "0.90-1.00"], include_lowest=True,
    )
    rng = np.random.default_rng(seed)
    groups = [g for _, g in frame.groupby(["risk_band", "Geography"], observed=True)]
    # proportional allocation, then top up from the largest strata
    picks = []
    for g in groups:
        k = max(1, round(n * len(g) / len(frame)))
        picks.append(g.sample(min(k, len(g)), random_state=int(rng.integers(1e6))))
    out = pd.concat(picks)
    if len(out) > n:
        out = out.sample(n, random_state=SEED)
    elif len(out) < n:
        rest = frame.drop(index=out.index)
        out = pd.concat([out, rest.sample(n - len(out), random_state=SEED)])
    return out.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def main():
    merged, X_C, y, test_idx = rebuild_hybrid_and_split()
    art = joblib.load(config.PROCESSED_DIR / "best_hybrid_xgb.joblib")
    pipe, names = art["pipe"], art["feature_names"]
    assert np.allclose(X_C[test_idx], art["X_test"]), "split does not match saved model"

    prob = pipe.predict_proba(art["X_test"])[:, 1]
    at_risk = prob >= RISK_THRESHOLD
    print(f"Held-out set {len(test_idx)}; at risk at p >= {RISK_THRESHOLD}: {at_risk.sum()}")

    raw = merged.iloc[test_idx].reset_index(drop=True)
    raw["churn_probability"] = prob
    raw["held_out_row"] = np.arange(len(raw))
    pool = raw[at_risk]
    if len(pool) < N_PROFILES:
        raise SystemExit(f"only {len(pool)} at-risk customers; need {N_PROFILES}")

    chosen = stratified_sample(pool, N_PROFILES)
    print(f"Sampled {len(chosen)} profiles "
          f"(p from {chosen.churn_probability.min():.3f} to {chosen.churn_probability.max():.3f})")

    # --- SHAP drivers, restricted to interpretable (non-embedding) features ---
    rows_local = chosen["held_out_row"].to_numpy()
    X_pre = pipe.named_steps["scaler"].transform(art["X_test"][rows_local])
    sv = shap.TreeExplainer(pipe.named_steps["clf"])(X_pre).values
    interp = [i for i, nm in enumerate(names) if not nm.startswith(("st_emb_", "bert_"))]
    print(f"Interpretable features available for driver phrasing: {len(interp)}")

    records = []
    for r, (_, cust) in enumerate(chosen.iterrows()):
        contrib = [(names[i], sv[r, i]) for i in interp]
        contrib.sort(key=lambda kv: abs(kv[1]), reverse=True)
        drivers, phrases = [], []
        for nm, val in contrib[:TOP_K_DRIVERS]:
            raw_val = cust[nm] if nm in cust.index else np.nan
            drivers.append(f"{nm}={raw_val}|shap={val:+.4f}")
            phrases.append(retention_llm.driver_to_phrase(nm, raw_val))
        blurb = retention_llm.profile_blurb(cust)
        records.append({
            "profile_id": f"P{r + 1:02d}",
            "held_out_row": int(cust["held_out_row"]),
            "churn_probability": round(float(cust["churn_probability"]), 4),
            "actual_exited": int(cust[config.TARGET_COL]),
            "geography": cust["Geography"],
            "profile": blurb,
            "top_drivers": " ; ".join(phrases),
            "driver_detail": " ; ".join(drivers),
            "prompt": retention_llm.build_prompt(blurb, phrases, POLICY),
            "message_generic": retention_llm.build_generic_message(),
            "message_llm": "",          # filled by stage 2
        })

    out = pd.DataFrame(records)
    path = config.PROCESSED_DIR / "retention_profiles.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote {path}  ({len(out)} profiles)")
    print("\nExample profile:")
    print(" ", out.loc[0, "profile"])
    print("  drivers:", out.loc[0, "top_drivers"])
    print("\nNext: stage 2, scripts/run_retention_generate.py (needs a generative model).")


if __name__ == "__main__":
    main()

"""Stage 4: score the completed rating sheets and produce the results table.

Un-blinds the sheets against the key, aggregates per dimension, measures
inter-rater reliability with Krippendorff's alpha on the interval scale, and
compares generic against generated messages with a paired test on the
per-profile rater means, correcting across the four dimensions.

Input :  data/processed/rating_sheets/rater_*.csv   (completed by the panel)
         data/processed/rating_key.csv
Output:  data/processed/table_retention_eval.csv    the results table
         data/processed/retention_eval_long.csv     tidy per-rating records
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from code import config, evaluate

DIMENSIONS = ["Relevance", "Personalization", "Actionability", "Persuasiveness"]


def krippendorff_alpha_interval(matrix: np.ndarray) -> float:
    """Krippendorff's alpha for interval data, tolerating missing cells.

    `matrix` is (raters x items) with np.nan for a missing rating. Implemented
    directly so the pipeline carries no extra dependency; agrees with the
    reference implementation on complete and incomplete designs.
    """
    m = np.asarray(matrix, dtype=float)
    n_per_item = np.sum(~np.isnan(m), axis=0)
    usable = n_per_item >= 2
    m, n_per_item = m[:, usable], n_per_item[usable]
    if m.size == 0:
        return float("nan")
    total = float(n_per_item.sum())

    # observed disagreement: mean squared difference within items
    d_o = 0.0
    for j in range(m.shape[1]):
        vals = m[~np.isnan(m[:, j]), j]
        nj = len(vals)
        d_o += np.sum((vals[:, None] - vals[None, :]) ** 2) / (nj - 1)
    d_o /= total

    # expected disagreement: mean squared difference across the whole pool
    pool = m[~np.isnan(m)]
    d_e = np.sum((pool[:, None] - pool[None, :]) ** 2) / (total * (total - 1))
    return float(1.0 - d_o / d_e) if d_e else float("nan")


def main():
    sheets = sorted((config.PROCESSED_DIR / "rating_sheets").glob("rater_*.csv"))
    key_path = config.PROCESSED_DIR / "rating_key.csv"
    if not sheets or not key_path.exists():
        raise SystemExit("No rating sheets or key found. Run stage 3 first.")
    key = pd.read_csv(key_path)

    frames = []
    for path in sheets:
        k = int(path.stem.split("_")[1])
        s = pd.read_csv(path)
        s["rater"] = k
        missing = s[DIMENSIONS].isna().any(axis=1).sum()
        if missing:
            print(f"  ! {path.name}: {missing} rows with a blank score (excluded)")
        frames.append(s)
    long = pd.concat(frames, ignore_index=True).merge(
        key, on=["rater", "profile_id", "message_label"], how="left", validate="many_to_one"
    )
    if long["kind"].isna().any():
        raise SystemExit("Some rows could not be matched to the key; sheets were altered.")
    for d in DIMENSIONS:
        long[d] = pd.to_numeric(long[d], errors="coerce")
        bad = long[d].dropna()
        if len(bad) and (bad.min() < 1 or bad.max() > 5):
            raise SystemExit(f"{d} has values outside 1-5.")
    long.to_csv(config.PROCESSED_DIR / "retention_eval_long.csv", index=False)

    n_raters = long["rater"].nunique()
    n_prof = long["profile_id"].nunique()
    scored = long[DIMENSIONS].notna().all(axis=1).sum()
    print(f"{len(long)} rows from {n_raters} raters over {n_prof} profiles; "
          f"{scored} fully scored\n")

    if scored == 0:
        raise SystemExit(
            "None of the rows carry scores yet, so there is nothing to analyse.\n"
            "\n"
            "The sheets in data/processed/rating_sheets/ are blank templates.\n"
            "Each rater must open their own rater_<k>.csv and enter a whole\n"
            "number from 1 to 5 in all four score columns for every row, then\n"
            "save the file in place. See rating_sheets/README.txt for the\n"
            "instructions and the 1-to-5 anchors.\n"
            "\n"
            "Re-run this script once at least one sheet has been completed."
        )

    # Every profile needs both a generic and a generated rating to be paired.
    kinds = set(long.loc[long[DIMENSIONS].notna().any(axis=1), "kind"])
    if not {"generic", "llm"} <= kinds:
        raise SystemExit(
            f"Only found ratings of kind {sorted(kinds)}. Both 'generic' and "
            "'llm' messages must be scored for the paired comparison; check "
            "that whole sheets were completed rather than parts of them."
        )

    rows, p_raw = [], []
    for d in DIMENSIONS:
        # per-profile mean across raters, so the paired unit is the profile
        piv = (long.pivot_table(index="profile_id", columns="kind", values=d, aggfunc="mean")
                   .dropna())
        gen, llm = piv["generic"].to_numpy(), piv["llm"].to_numpy()
        t, p_t = evaluate.paired_t_test(llm, gen)
        _, p_w = evaluate.wilcoxon_test(llm, gen)
        diff = llm - gen
        ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(),
                              scale=stats.sem(diff)) if len(diff) > 1 else (np.nan, np.nan)
        # reliability on the pooled ratings for this dimension
        mat = long.pivot_table(index="rater", columns=["profile_id", "message_label"],
                               values=d, aggfunc="mean").to_numpy()
        rows.append({
            "dimension": d,
            "generic_mean": round(gen.mean(), 3), "generic_sd": round(gen.std(ddof=1), 3),
            "llm_mean": round(llm.mean(), 3), "llm_sd": round(llm.std(ddof=1), 3),
            "delta": round(diff.mean(), 3),
            "delta_ci95_low": round(ci[0], 3), "delta_ci95_high": round(ci[1], 3),
            "t": round(t, 3), "p_t": p_t, "p_wilcoxon": p_w,
            "krippendorff_alpha": round(krippendorff_alpha_interval(mat), 3),
            "n_profiles": len(diff),
        })
        p_raw.append(p_t)

    flags = evaluate.benjamini_hochberg(p_raw, fdr=0.05)
    for r, f in zip(rows, flags):
        r["bh_significant"] = bool(f)
        r["p_t"] = round(r["p_t"], 6)
        r["p_wilcoxon"] = round(r["p_wilcoxon"], 6)

    table = pd.DataFrame(rows)
    overall = long.pivot_table(index="profile_id", columns="kind",
                               values=DIMENSIONS, aggfunc="mean").dropna()
    g_all = overall.xs("generic", level=1, axis=1).mean(axis=1)
    l_all = overall.xs("llm", level=1, axis=1).mean(axis=1)
    table.loc[len(table)] = {
        "dimension": "Mean of four dimensions",
        "generic_mean": round(g_all.mean(), 3), "generic_sd": round(g_all.std(ddof=1), 3),
        "llm_mean": round(l_all.mean(), 3), "llm_sd": round(l_all.std(ddof=1), 3),
        "delta": round((l_all - g_all).mean(), 3),
        "delta_ci95_low": np.nan, "delta_ci95_high": np.nan,
        "t": np.nan, "p_t": np.nan, "p_wilcoxon": np.nan,
        "krippendorff_alpha": np.nan, "n_profiles": len(g_all), "bh_significant": None,
    }

    out = config.PROCESSED_DIR / "table_retention_eval.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {out}")
    print("\nTranscribe these values into Table 15 of Dissertation.md, replace the "
          "'EVALUATION PENDING' wording in Section 4.3.8, then rebuild.")


if __name__ == "__main__":
    main()

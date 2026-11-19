"""Lightweight complaint feature extraction (sentiment + severity + LDA) on the
100K CFPB sample, then aggregate into the fusion table.

Output:
    data/processed/cfpb_enriched.parquet
    data/processed/fusion_table.csv
    data/processed/bank_merged.parquet  (bank rows + joined complaint features)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from code import config, complaint_features, data_loader, fusion, preprocess


def main():
    cfpb = pd.read_parquet(config.PROCESSED_DIR / "cfpb_sample.parquet")
    print(f"Loaded {len(cfpb):,} CFPB rows; columns: {list(cfpb.columns)}")

    # Sentiment + severity + LDA
    print("Computing sentiment + severity + LDA topics...")
    cfpb = complaint_features.enrich(cfpb)

    out_enriched = config.PROCESSED_DIR / "cfpb_enriched.parquet"
    cfpb.to_parquet(out_enriched, index=False)
    print(f"Wrote {out_enriched}")
    print(cfpb[["sentiment", "severity", "dominant_topic"]].describe())

    ftable = fusion.aggregate_complaint_features(cfpb)
    out_ftable = config.PROCESSED_DIR / "fusion_table.csv"
    ftable.to_csv(out_ftable, index=False)
    print(f"\nFusion table:\n{ftable}")

    bank = preprocess.clean_bank_churn(data_loader.load_bank_churn())
    merged = fusion.join_to_bank(bank, ftable)
    out_merged = config.PROCESSED_DIR / "bank_merged.parquet"
    merged.to_parquet(out_merged, index=False)
    print(f"\nMerged shape: {merged.shape}")
    print("Sample of joined complaint features:")
    print(merged[["product_bucket", "region", "avg_sentiment", "avg_severity",
                  "complaint_rate_per_1k", "dominant_topic"]].head(10))


if __name__ == "__main__":
    main()

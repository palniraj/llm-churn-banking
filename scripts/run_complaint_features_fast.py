"""Fast (CPU-friendly) complaint feature extraction.

Substitutes the DistilBERT sentiment classifier for VADER lexicon-based polarity
to keep wall-clock cost manageable on CPU. The dissertation lists this as an
acceptable approximation when GPU sentiment classification is unavailable;
the substitution is documented in §4.2.4.

Output:
    data/processed/cfpb_enriched.parquet
    data/processed/fusion_table.csv
    data/processed/bank_merged.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from code import complaint_features, config, data_loader, fusion, preprocess


def vader_sentiment(narratives) -> np.ndarray:
    """Compute VADER compound polarity in [-1, +1]."""
    import nltk
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download("vader_lexicon", quiet=True)
    sid = SentimentIntensityAnalyzer()
    out = np.zeros(len(narratives), dtype=np.float32)
    for i, t in enumerate(narratives):
        if not t:
            out[i] = 0.0
            continue
        out[i] = sid.polarity_scores(t)["compound"]
        if i % 20_000 == 0:
            print(f"  sentiment: {i:,} / {len(narratives):,}")
    return out


def main():
    cfpb = pd.read_parquet(config.PROCESSED_DIR / "cfpb_sample.parquet")
    print(f"Loaded {len(cfpb):,} CFPB rows")

    # Truncate narratives so LDA + sentiment scale linearly
    cfpb["narrative"] = cfpb["narrative"].astype(str).str.slice(0, 2000)

    print("Computing VADER sentiment polarity...")
    pol = vader_sentiment(cfpb["narrative"].tolist())
    cfpb["sentiment"] = pol

    print("Computing severity score...")
    cfpb["severity"] = complaint_features.severity_score(cfpb["narrative"], polarity=pol)

    print("Fitting LDA (10 topics, 5 passes)...")
    _, _, M = complaint_features.fit_lda_topics(
        cfpb["narrative"], num_topics=10, passes=5
    )
    for k in range(M.shape[1]):
        cfpb[f"topic_{k}"] = M[:, k]
    cfpb["dominant_topic"] = M.argmax(axis=1)

    out_enriched = config.PROCESSED_DIR / "cfpb_enriched.parquet"
    cfpb.to_parquet(out_enriched, index=False)
    print(f"\nWrote {out_enriched}")
    print(cfpb[["sentiment", "severity", "dominant_topic"]].describe())

    ftable = fusion.aggregate_complaint_features(cfpb)
    out_ftable = config.PROCESSED_DIR / "fusion_table.csv"
    ftable.to_csv(out_ftable, index=False)
    print(f"\nFusion table ({len(ftable)} keys):")
    print(ftable.to_string(index=False))

    bank = preprocess.clean_bank_churn(data_loader.load_bank_churn())
    merged = fusion.join_to_bank(bank, ftable)
    out_merged = config.PROCESSED_DIR / "bank_merged.parquet"
    merged.to_parquet(out_merged, index=False)
    print(f"\nMerged shape: {merged.shape}")
    print(merged[["product_bucket", "region", "avg_sentiment", "avg_severity",
                  "complaint_rate_per_1k", "dominant_topic"]].head(10))


if __name__ == "__main__":
    main()

"""Stream-load the CFPB CSV, filter to banking-related complaints with non-empty
narratives, and write a stratified 100K sample to disk as parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from code import config, data_loader, preprocess


def main():
    cfpb = data_loader.load_cfpb_complaints()
    print(f"Banking subset with non-empty narrative: {len(cfpb):,}")

    cfpb["narrative"] = cfpb["narrative"].apply(preprocess.clean_narrative)
    sample = data_loader.stratified_sample(cfpb, by="product", n=config.CFPB_SAMPLE_SIZE)
    print(f"Stratified sample: {len(sample):,}")
    print(sample["product"].value_counts())

    out = config.PROCESSED_DIR / "cfpb_sample.parquet"
    sample.to_parquet(out, index=False)
    print(f"Wrote {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()

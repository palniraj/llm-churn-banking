"""Probabilistic data fusion: aggregate complaint features by (product_bucket, region)
and join back to the structured churn dataset.

The fusion key is (product_bucket, region), giving 20 populated keys (four product
buckets by five regions). Consumer age is not published in the CFPB extract, so it
does not participate in the key; see Section 3.3 of the dissertation.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Map CFPB product -> bank-product proxy used to align with the Kaggle bank dataset.
# The Kaggle bank dataset has no explicit product column, so we use NumOfProducts buckets.
PRODUCT_BUCKET_MAP = {
    "Checking or savings account": "checking_savings",
    "Credit card or prepaid card": "credit_card",
    "Credit reporting, credit repair services, or other personal consumer reports": "credit_report",
    "Mortgage": "mortgage",
}

# US state -> region (rough mapping; non-US records fall through to "OTHER")
_STATE_TO_REGION = {
    "WA": "WEST", "OR": "WEST", "CA": "WEST", "NV": "WEST", "AZ": "WEST",
    "ID": "WEST", "MT": "WEST", "WY": "WEST", "UT": "WEST", "CO": "WEST",
    "NM": "WEST", "AK": "WEST", "HI": "WEST",
    "ND": "MIDWEST", "SD": "MIDWEST", "NE": "MIDWEST", "KS": "MIDWEST",
    "MN": "MIDWEST", "IA": "MIDWEST", "MO": "MIDWEST", "WI": "MIDWEST",
    "IL": "MIDWEST", "IN": "MIDWEST", "MI": "MIDWEST", "OH": "MIDWEST",
    "TX": "SOUTH", "OK": "SOUTH", "AR": "SOUTH", "LA": "SOUTH",
    "MS": "SOUTH", "AL": "SOUTH", "TN": "SOUTH", "KY": "SOUTH",
    "GA": "SOUTH", "FL": "SOUTH", "SC": "SOUTH", "NC": "SOUTH",
    "VA": "SOUTH", "WV": "SOUTH",
    "PA": "NORTHEAST", "NY": "NORTHEAST", "NJ": "NORTHEAST",
    "CT": "NORTHEAST", "RI": "NORTHEAST", "MA": "NORTHEAST",
    "VT": "NORTHEAST", "NH": "NORTHEAST", "ME": "NORTHEAST",
    "DC": "NORTHEAST", "MD": "NORTHEAST", "DE": "NORTHEAST",
}

# Bank dataset Geography -> region proxy
_GEO_TO_REGION = {"France": "WEST", "Germany": "MIDWEST", "Spain": "SOUTH"}


def _state_to_region(state: str) -> str:
    return _STATE_TO_REGION.get(str(state).upper(), "OTHER")


def aggregate_complaint_features(
    cfpb: pd.DataFrame,
    age_band_col: str = "age_band",
    state_col: str = "state",
) -> pd.DataFrame:
    """Group complaints by (product_bucket, region) and compute aggregate signals.

    CFPB does not publish consumer age, so the key is (product_bucket, region) only.
    `age_band_col` is accepted but unused, and is retained so that the key can be
    extended without a signature change if an age-bearing complaint source is wired in.
    """
    df = cfpb.copy()
    df["product_bucket"] = df["product"].map(PRODUCT_BUCKET_MAP).fillna("other")
    df["region"] = df[state_col].map(_state_to_region) if state_col in df.columns else "OTHER"

    agg = df.groupby(["product_bucket", "region"]).agg(
        avg_sentiment=("sentiment", "mean"),
        avg_severity=("severity", "mean"),
        complaint_count=("narrative", "size"),
        dominant_topic=("dominant_topic", lambda s: int(s.mode().iat[0]) if len(s) else -1),
    ).reset_index()
    # complaints per 1k records (rate normalisation)
    total = agg["complaint_count"].sum() or 1
    agg["complaint_rate_per_1k"] = 1000.0 * agg["complaint_count"] / total
    return agg


def _bank_product_bucket(num_products: int) -> str:
    """Map NumOfProducts in the Kaggle bank dataset to a coarse product bucket."""
    if num_products >= 3:
        return "credit_card"  # multi-product holders skew towards card-led portfolios
    if num_products == 2:
        return "checking_savings"
    return "credit_report"


def join_to_bank(
    bank: pd.DataFrame,
    fusion_table: pd.DataFrame,
    geography_col: str = "Geography",
    num_products_col: str = "NumOfProducts",
) -> pd.DataFrame:
    """Join the aggregate fusion table back onto each bank-customer row."""
    df = bank.copy()
    df["product_bucket"] = df[num_products_col].astype(int).map(_bank_product_bucket)
    df["region"] = df[geography_col].map(_GEO_TO_REGION).fillna("OTHER")

    merged = df.merge(fusion_table, on=["product_bucket", "region"], how="left")
    # Fall back to global average when key is unmatched
    for col in ("avg_sentiment", "avg_severity", "complaint_rate_per_1k"):
        if col in merged.columns:
            merged[col] = merged[col].fillna(fusion_table[col].mean())
    if "dominant_topic" in merged.columns:
        merged["dominant_topic"] = merged["dominant_topic"].fillna(-1).astype(int)
    return merged


def get_complaint_feature_columns() -> list[str]:
    """Names of the complaint-derived features added by `join_to_bank`."""
    return [
        "avg_sentiment",
        "avg_severity",
        "complaint_rate_per_1k",
        "dominant_topic",
    ]

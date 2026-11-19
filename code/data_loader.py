"""Dataset loading utilities for Bank Customer Churn and CFPB Complaints."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def load_bank_churn(path: Path | None = None) -> pd.DataFrame:
    """Load the Kaggle Bank Customer Churn dataset (10,000 records, 14 features)."""
    path = Path(path or config.BANK_CHURN_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"Bank Churn CSV not found at {path}. "
            f"Download from https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction "
            f"and place it as {path}."
        )
    df = pd.read_csv(path)
    drop = [c for c in config.DROP_COLS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    logger.info("Loaded bank churn: %s rows, %s cols", len(df), df.shape[1])
    return df


def load_maven_churn(path: Path | None = None) -> pd.DataFrame:
    """Load the out-of-distribution Bank Churn validation dataset.

    The default path points at the Kaggle Playground Series S4E1 raw training
    set, which shares the canonical schema with `bank_churn.csv` (just `id`
    instead of `RowNumber`) and has ~165K rows from a different sample.
    """
    path = Path(path or config.MAVEN_CHURN_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"OOD Bank Churn CSV not found at {path}. "
            f"Run scripts/download_data.sh to fetch it."
        )
    df = pd.read_csv(path)
    # Normalise the identifier column so downstream code is schema-agnostic
    if "id" in df.columns and "RowNumber" not in df.columns:
        df = df.rename(columns={"id": "RowNumber"})
    drop = [c for c in config.DROP_COLS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    return df


def load_cfpb_complaints(
    path: Path | None = None,
    chunksize: int = 50_000,
    products: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Stream-load CFPB complaints, keep banking-related rows with non-empty narrative."""
    path = Path(path or config.CFPB_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"CFPB CSV not found at {path}. "
            f"Download from https://www.consumerfinance.gov/data-research/consumer-complaints/ "
            f"and place it as {path}."
        )
    products = tuple(products) if products is not None else config.BANKING_PRODUCTS

    keep_cols = [
        "Date received",
        "Product",
        "Sub-product",
        "Issue",
        "Consumer complaint narrative",
        "Company",
        "State",
        "Company response to consumer",
        "Timely response?",
        "Consumer disputed?",
    ]

    chunks = []
    for i, chunk in enumerate(
        pd.read_csv(path, usecols=lambda c: c in keep_cols, chunksize=chunksize, low_memory=False)
    ):
        chunk = chunk.dropna(subset=["Consumer complaint narrative"])
        chunk = chunk[chunk["Product"].isin(products)]
        chunks.append(chunk)
        if i % 10 == 0:
            logger.info("CFPB chunk %d: kept %d rows so far", i, sum(len(c) for c in chunks))

    df = pd.concat(chunks, ignore_index=True)
    df = df.rename(
        columns={
            "Date received": "date_received",
            "Product": "product",
            "Sub-product": "sub_product",
            "Issue": "issue",
            "Consumer complaint narrative": "narrative",
            "Company": "company",
            "State": "state",
            "Company response to consumer": "company_response",
            "Timely response?": "timely_response",
            "Consumer disputed?": "consumer_disputed",
        }
    )
    logger.info("Loaded CFPB banking subset: %d rows", len(df))
    return df


def stratified_sample(df: pd.DataFrame, by: str, n: int, seed: int = config.PRIMARY_SEED) -> pd.DataFrame:
    """Return a stratified sample of `n` rows from `df` by column `by`."""
    if len(df) <= n:
        return df.copy()
    fraction = n / len(df)
    sampled = df.groupby(by, group_keys=False).apply(
        lambda g: g.sample(n=max(1, int(round(len(g) * fraction))), random_state=seed)
    )
    return sampled.reset_index(drop=True)

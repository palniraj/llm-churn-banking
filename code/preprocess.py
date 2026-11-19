"""Pre-processing for the Bank Customer Churn dataset and CFPB complaint narratives."""
from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config

# ---------------------------------------------------------------------------
# Bank Churn
# ---------------------------------------------------------------------------
def clean_bank_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Type-fix, median-impute, and add derived columns for downstream pipelines."""
    df = df.copy()
    for col in config.NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
    for col in config.BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    df["age_band"] = pd.cut(
        df["Age"],
        bins=[0, 24, 34, 44, 54, 200],
        labels=["<25", "25-34", "35-44", "45-54", "55+"],
    ).astype(str)
    return df


def split_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y). X excludes the target and any text/aux columns."""
    y = df[config.TARGET_COL].astype(int)
    drop = [config.TARGET_COL]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    return X, y


def encode_traditional(X: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """One-hot encode categoricals, scale numerics, return (matrix, feature_names)."""
    X = X.copy()
    # one-hot
    cat_cols = [c for c in config.CATEGORICAL_COLS if c in X.columns]
    X_oh = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    # drop helper columns that are not features
    for col in ("age_band",):
        if col in X_oh.columns:
            X_oh = X_oh.drop(columns=[col])
    # cast all columns to float so StandardScaler can mutate them in-place
    X_oh = X_oh.astype(float)
    # scale numerics
    num_cols = [c for c in config.NUMERIC_COLS if c in X_oh.columns]
    scaler = StandardScaler()
    X_oh.loc[:, num_cols] = scaler.fit_transform(X_oh[num_cols])
    return X_oh, list(X_oh.columns)


# ---------------------------------------------------------------------------
# CFPB narratives
# ---------------------------------------------------------------------------
_REDACT = re.compile(r"X{2,}|\[redacted\]", re.IGNORECASE)
_WS = re.compile(r"\s+")


def clean_narrative(text: str) -> str:
    """Lower-case, remove redaction tokens, collapse whitespace, trim."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _REDACT.sub("[redacted]", text)
    text = _WS.sub(" ", text).strip()
    return text


def add_age_band_column(df: pd.DataFrame, age_col: str = "Age") -> pd.DataFrame:
    df = df.copy()
    df["age_band"] = pd.cut(
        df[age_col],
        bins=[0, 24, 34, 44, 54, 200],
        labels=["<25", "25-34", "35-44", "45-54", "55+"],
    ).astype(str)
    return df

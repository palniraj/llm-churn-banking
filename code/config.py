"""Project-wide configuration constants.

All paths are relative to the thesis/ directory.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT / "figures"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BANK_CHURN_CSV = DATA_DIR / "bank_churn.csv"
MAVEN_CHURN_CSV = DATA_DIR / "maven_bank_churn.csv"
CFPB_CSV = DATA_DIR / "cfpb_complaints.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEEDS = (42, 123, 2024)
PRIMARY_SEED = 42

# ---------------------------------------------------------------------------
# Splits & cross-validation
# ---------------------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # of the original dataset
N_SPLITS_CV = 10

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
BERT_MODEL_NAME = "bert-base-uncased"
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# ---------------------------------------------------------------------------
# Complaint analysis
# ---------------------------------------------------------------------------
LDA_NUM_TOPICS = 10
CFPB_SAMPLE_SIZE = 100_000
BANKING_PRODUCTS = (
    "Checking or savings account",
    "Credit card or prepaid card",
    "Credit reporting, credit repair services, or other personal consumer reports",
    "Mortgage",
)

# ---------------------------------------------------------------------------
# Retention generation
# ---------------------------------------------------------------------------
AT_RISK_THRESHOLD = 0.7
RETENTION_TEMPERATURE = 0.7
RETENTION_TOP_P = 0.9
RETENTION_MAX_TOKENS = 200

# ---------------------------------------------------------------------------
# Feature columns of the Bank Customer Churn dataset
# ---------------------------------------------------------------------------
NUMERIC_COLS = (
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
)
BINARY_COLS = ("HasCrCard", "IsActiveMember")
CATEGORICAL_COLS = ("Geography", "Gender")
TARGET_COL = "Exited"
DROP_COLS = ("RowNumber", "CustomerId", "Surname")

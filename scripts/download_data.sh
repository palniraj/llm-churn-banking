#!/usr/bin/env bash
# Download the three datasets used by the dissertation pipeline.
#   1. Bank Customer Churn (Kaggle, 10K rows)        -> data/bank_churn.csv
#   2. Bank Churn S4E1 raw (Kaggle, 165K rows, OOD)  -> data/maven_bank_churn.csv
#   3. CFPB Consumer Complaints (CFPB direct, 1.8GB) -> data/cfpb_complaints.csv
#
# Prerequisites:
#   - Kaggle API token at ~/.kaggle/access_token (recommended) or ~/.kaggle/kaggle.json
#   - Python 3 + venv module
#   - curl, unzip
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

# --- Set up Kaggle CLI in a local virtualenv ------------------------------
if [[ ! -d .venv-data ]]; then
  python3 -m venv .venv-data
  .venv-data/bin/pip install --quiet --upgrade pip kaggle
fi
KAGGLE=.venv-data/bin/kaggle

# --- 1. Primary dataset ---------------------------------------------------
if [[ ! -f data/bank_churn.csv ]]; then
  echo "[1/3] Downloading Bank Customer Churn (Kaggle)..."
  $KAGGLE datasets download -d shantanudhakadd/bank-customer-churn-prediction \
      -p data/ --unzip
  mv -f data/Churn_Modelling.csv data/bank_churn.csv
fi

# --- 2. OOD validation dataset -------------------------------------------
if [[ ! -f data/maven_bank_churn.csv ]]; then
  echo "[2/3] Downloading Bank Churn S4E1 raw (Kaggle, OOD validation)..."
  $KAGGLE datasets download -d yatharthgautam123789/playground-series-s4e1 \
      -p data/ --unzip
  mv -f data/train.csv data/maven_bank_churn.csv
  rm -f data/test.csv data/sample_submission.csv
fi

# --- 3. CFPB Consumer Complaints -----------------------------------------
if [[ ! -f data/cfpb_complaints.csv ]]; then
  echo "[3/3] Downloading CFPB Consumer Complaints (~1.8 GB, can take 5-15 min)..."
  curl -L -o data/cfpb_complaints.csv.zip \
       "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
  echo "Unzipping..."
  unzip -o data/cfpb_complaints.csv.zip -d data/
  if [[ -f data/complaints.csv ]]; then
    mv -f data/complaints.csv data/cfpb_complaints.csv
  fi
  rm -f data/cfpb_complaints.csv.zip
fi

echo
echo "All datasets downloaded:"
ls -lh data/*.csv

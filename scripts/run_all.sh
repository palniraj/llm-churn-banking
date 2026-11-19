#!/usr/bin/env bash
# End-to-end pipeline: data -> embeddings -> fusion -> classifiers -> SHAP -> retention -> figures.
# Run from the thesis/ directory.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. EDA
jupyter nbconvert --to notebook --execute notebooks/01_data_eda.ipynb --output 01_data_eda.executed.ipynb

# 2. Tabular -> text + embeddings
jupyter nbconvert --to notebook --execute notebooks/02_tabular_to_text_embeddings.ipynb --output 02_tabular_to_text_embeddings.executed.ipynb

# 3. Complaint analysis + fusion
jupyter nbconvert --to notebook --execute notebooks/03_complaint_analysis_fusion.ipynb --output 03_complaint_analysis_fusion.executed.ipynb

# 4. Classifier training across configs A/B/C  (writes tables + figures)
jupyter nbconvert --to notebook --execute notebooks/04_classifiers_configs_ABC.ipynb --output 04_classifiers_configs_ABC.executed.ipynb

# 5. SHAP explainability
jupyter nbconvert --to notebook --execute notebooks/05_shap_explainability.ipynb --output 05_shap_explainability.executed.ipynb

# 6. Retention message generation + human-eval aggregation
jupyter nbconvert --to notebook --execute notebooks/06_retention_message_generation.ipynb --output 06_retention_message_generation.executed.ipynb

# 7. Refresh figure placeholders (only those not produced by notebooks)
python scripts/make_figures.py

echo "Pipeline complete. Tables in data/processed/*.csv, figures in figures/*.png."

"""Render bank-churn rows to text and embed them with Sentence-Transformer.

Output:
    data/processed/bank_emb_st.npy   (10000, 384) float32
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from code import config, data_loader, embeddings, preprocess, tabular_to_text


def main():
    bank = preprocess.clean_bank_churn(data_loader.load_bank_churn())
    texts = tabular_to_text.render(bank, template="declarative")
    print(f"Rendered {len(texts):,} customer descriptions; first row:")
    print(" ", texts.iloc[0])

    emb = embeddings.embed_sentence_transformer(texts, batch_size=64, device="cpu")
    print(f"Embeddings shape: {emb.shape}")
    embeddings.cache_embeddings(emb, "bank_emb_st")


if __name__ == "__main__":
    main()

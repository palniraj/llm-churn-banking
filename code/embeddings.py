"""LLM embedding generation: BERT [CLS] and Sentence-Transformers mean-pool."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def embed_sentence_transformer(
    texts: pd.Series, batch_size: int = 64, device: str | None = None
) -> np.ndarray:
    """Embed texts with sentence-transformers/all-MiniLM-L6-v2 (384 dim)."""
    from sentence_transformers import SentenceTransformer
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Sentence-Transformer embedding %d texts on %s", len(texts), device)
    model = SentenceTransformer(config.ST_MODEL_NAME, device=device)
    emb = model.encode(
        texts.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def embed_bert_cls(
    texts: pd.Series,
    batch_size: int = 32,
    device: str | None = None,
    max_length: int = 512,
) -> np.ndarray:
    """Embed texts with bert-base-uncased; return [CLS] from the last hidden layer (768 dim)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(config.BERT_MODEL_NAME).to(device).eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts.iloc[i : i + batch_size].tolist()
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=device.startswith("cuda")):
                h = model(**enc).last_hidden_state  # (B, T, H)
            cls = h[:, 0, :].float().cpu().numpy()
            out.append(cls)
            if (i // batch_size) % 20 == 0:
                logger.info("BERT batch %d / %d", i // batch_size, len(texts) // batch_size)
    return np.vstack(out).astype(np.float32)


def cache_embeddings(arr: np.ndarray, name: str) -> Path:
    """Persist embeddings to data/processed/<name>.npy and return the path."""
    path = config.PROCESSED_DIR / f"{name}.npy"
    np.save(path, arr)
    logger.info("Cached embeddings to %s (shape=%s)", path, arr.shape)
    return path


def load_cached(name: str) -> np.ndarray | None:
    """Return cached embeddings if present, else None."""
    path = config.PROCESSED_DIR / f"{name}.npy"
    if path.exists():
        return np.load(path)
    return None

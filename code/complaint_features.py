"""Per-complaint feature extraction: sentiment polarity, severity, LDA topic distribution."""
from __future__ import annotations

import logging
import re
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

_ESCALATION_KEYWORDS: Tuple[str, ...] = (
    "fraud", "fraudulent", "stolen", "unauthorized", "unauthorised",
    "scam", "lawsuit", "attorney", "legal action", "discrimination",
    "harassment", "threat", "predatory", "deceptive",
)


# ---------------------------------------------------------------------------
# Sentiment polarity in [-1, +1]
# ---------------------------------------------------------------------------
def sentiment_polarity(narratives: Iterable[str], batch_size: int = 32) -> np.ndarray:
    """Return polarity in [-1, +1] using DistilBERT-SST2."""
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "sentiment-analysis",
        model=config.SENTIMENT_MODEL_NAME,
        device=device,
        truncation=True,
        max_length=512,
    )
    texts = list(narratives)
    out = np.zeros(len(texts), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        preds = pipe(chunk)
        for j, p in enumerate(preds):
            score = float(p["score"])
            out[i + j] = score if p["label"].upper() == "POSITIVE" else -score
        if (i // batch_size) % 50 == 0:
            logger.info("Sentiment batch %d / %d", i // batch_size, len(texts) // batch_size)
    return out


# ---------------------------------------------------------------------------
# Severity score in [1, 5]
# ---------------------------------------------------------------------------
_KEYWORD_RE = re.compile("|".join(rf"\b{k}\b" for k in _ESCALATION_KEYWORDS), re.IGNORECASE)


def severity_score(narratives: Iterable[str], polarity: np.ndarray | None = None) -> np.ndarray:
    """Heuristic severity in [1, 5]: combines |polarity|, length, escalation keywords, capitalisation."""
    texts = list(narratives)
    if polarity is None:
        polarity = sentiment_polarity(texts)
    out = np.zeros(len(texts), dtype=np.float32)
    for i, t in enumerate(texts):
        if not t:
            out[i] = 1.0
            continue
        # |polarity| in [0, 1]
        p = float(abs(polarity[i]))
        # length in [0, 1] (saturates at 1000 chars)
        L = min(1.0, len(t) / 1000.0)
        # keyword density in [0, 1]
        keys = len(_KEYWORD_RE.findall(t))
        K = min(1.0, keys / 3.0)
        # caps ratio in [0, 1]
        words = t.split()
        caps = sum(1 for w in words if w.isupper() and len(w) > 1)
        C = min(1.0, caps / max(1, len(words)) * 3.0)
        # weighted aggregate -> [1, 5]
        raw = 0.45 * p + 0.20 * L + 0.25 * K + 0.10 * C
        out[i] = 1.0 + 4.0 * raw
    return out


# ---------------------------------------------------------------------------
# LDA topic distribution
# ---------------------------------------------------------------------------
def fit_lda_topics(
    narratives: Iterable[str], num_topics: int | None = None, passes: int = 10
):
    """Fit LDA on narratives and return (model, vectorizer, doc_topic_matrix).

    Uses scikit-learn's LatentDirichletAllocation by default (no gensim / numpy<2
    constraint, works on Colab's numpy 2.x). If gensim is explicitly requested via
    the environment, the historical gensim path is used instead.
    """
    import os

    num_topics = num_topics or config.LDA_NUM_TOPICS

    if os.environ.get("USE_GENSIM_LDA") == "1":
        return _fit_lda_gensim(narratives, num_topics, passes)

    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(
        stop_words="english", min_df=10, max_df=0.5, max_features=5000
    )
    X = vec.fit_transform([t if isinstance(t, str) else "" for t in narratives])
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        max_iter=max(5, passes),
        learning_method="online",
        random_state=config.PRIMARY_SEED,
        n_jobs=-1,
    )
    M = lda.fit_transform(X).astype(np.float32)
    return lda, vec, M


def _fit_lda_gensim(narratives, num_topics, passes):
    """Legacy gensim LDA path (requires numpy<2; enable with USE_GENSIM_LDA=1)."""
    from gensim import corpora, models
    from gensim.parsing.preprocessing import STOPWORDS, preprocess_string

    docs = [preprocess_string(t) for t in narratives]
    docs = [[w for w in d if w not in STOPWORDS and len(w) > 2] for d in docs]
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    bow = [dictionary.doc2bow(d) for d in docs]
    lda = models.LdaMulticore(
        bow, num_topics=num_topics, id2word=dictionary,
        passes=passes, workers=4, random_state=config.PRIMARY_SEED,
    )
    M = np.zeros((len(docs), num_topics), dtype=np.float32)
    for i, b in enumerate(bow):
        for k, p in lda.get_document_topics(b, minimum_probability=0.0):
            M[i, k] = p
    return lda, dictionary, M


# ---------------------------------------------------------------------------
# Driver: enrich CFPB df with per-complaint features
# ---------------------------------------------------------------------------
def enrich(df: pd.DataFrame, narrative_col: str = "narrative") -> pd.DataFrame:
    df = df.copy()
    pol = sentiment_polarity(df[narrative_col])
    df["sentiment"] = pol
    df["severity"] = severity_score(df[narrative_col], polarity=pol)
    _, _, M = fit_lda_topics(df[narrative_col])
    for k in range(M.shape[1]):
        df[f"topic_{k}"] = M[:, k]
    df["dominant_topic"] = M.argmax(axis=1)
    return df

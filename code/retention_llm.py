"""Personalized retention message generation and human-evaluation helpers."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Driver -> human-readable phrasing
# ---------------------------------------------------------------------------
DRIVER_PHRASES = {
    "Age": lambda v: f"customer age ({int(v)}) is in a band associated with elevated churn",
    "NumOfProducts": lambda v: f"holds only {int(v)} banking product(s)",
    "IsActiveMember": lambda v: "is currently flagged as inactive" if int(v) == 0 else "is an active member",
    "Balance": lambda v: ("account balance is zero" if float(v) <= 0 else f"current balance is ${float(v):,.0f}"),
    "Geography_Germany": lambda v: "based in Germany, where churn rates are higher in the dataset",
    "Gender_Male": lambda v: "male customer (Gender_Male encoded)",
    "CreditScore": lambda v: f"credit score of {int(v)}",
    "Tenure": lambda v: f"tenure of {int(v)} years",
    "EstimatedSalary": lambda v: f"estimated annual salary of ${float(v):,.0f}",
    "HasCrCard": lambda v: ("does not hold a credit card" if int(v) == 0 else "holds a credit card"),
    "avg_sentiment": lambda v: f"aggregated complaint sentiment in this product/region cluster is {float(v):+.2f}",
    "avg_severity": lambda v: f"aggregated complaint severity in this cluster is {float(v):.2f}/5",
    "complaint_rate_per_1k": lambda v: f"complaint rate is {float(v):.1f} per 1k customers in this cluster",
    "dominant_topic": lambda v: f"the prevailing complaint theme in this cluster is topic {int(v)} of the {config.LDA_TOPICS if hasattr(config, 'LDA_TOPICS') else 10}-topic model",
    "complaint_count": lambda v: f"{int(v)} complaints recorded for this product/region cluster",
    "Geography_Spain": lambda v: "based in Spain",
    "Geography_France": lambda v: "based in France",
}


def driver_to_phrase(name: str, value) -> str:
    fn = DRIVER_PHRASES.get(name)
    if fn is None:
        return f"feature {name} = {value}"
    try:
        return fn(value)
    except Exception:
        return f"feature {name} = {value}"


# ---------------------------------------------------------------------------
# Profile + prompt
# ---------------------------------------------------------------------------
def profile_blurb(row: pd.Series) -> str:
    """Compact natural-language customer profile for the prompt."""
    return (
        f"{int(row['Age'])}-year-old {str(row['Gender']).lower()} from {row['Geography']}, "
        f"credit score {int(row['CreditScore'])}, tenure {int(row['Tenure'])} years, "
        f"{int(row['NumOfProducts'])} product(s), "
        f"{'has' if int(row['HasCrCard']) else 'no'} credit card, "
        f"{'active' if int(row['IsActiveMember']) else 'inactive'} member, "
        f"balance ${float(row['Balance']):,.0f}, "
        f"estimated salary ${float(row['EstimatedSalary']):,.0f}."
    )


def build_prompt(profile: str, drivers: List[str], policy: str) -> str:
    return (
        "You are a senior customer-retention specialist at a retail bank. "
        "Write a short retention message (3-5 sentences) for the following customer.\n\n"
        f"Customer profile: {profile}\n"
        "Top reasons they may churn (from SHAP):\n- "
        + "\n- ".join(drivers)
        + f"\n\nBank retention policy: {policy}\n\n"
        "Constraints: do not invent products the bank does not offer, "
        "do not promise specific monetary amounts, "
        "use a warm but professional tone. "
        "Address the customer politely (e.g., 'Dear ...'). "
        "End with a clear next-step call-to-action."
    )


def build_generic_message() -> str:
    return (
        "Dear Customer, we noticed you may be considering leaving. "
        "We value your business. Please contact us for special offers."
    )


# ---------------------------------------------------------------------------
# LLM generation backends
# ---------------------------------------------------------------------------
@dataclass
class GenerationConfig:
    temperature: float = config.RETENTION_TEMPERATURE
    top_p: float = config.RETENTION_TOP_P
    max_new_tokens: int = config.RETENTION_MAX_TOKENS


def generate_with_hf(prompt: str, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                     gen: GenerationConfig | None = None) -> str:
    """Generate via HuggingFace transformers (with optional bitsandbytes 4-bit)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gen = gen or GenerationConfig()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=gen.max_new_tokens,
        do_sample=True,
        temperature=gen.temperature,
        top_p=gen.top_p,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def generate_with_openai(prompt: str, model: str = "gpt-4o-mini",
                         gen: GenerationConfig | None = None) -> str:
    """Generate via the OpenAI API. Requires OPENAI_API_KEY in the environment."""
    from openai import OpenAI

    gen = gen or GenerationConfig()
    client = OpenAI()
    rsp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=gen.temperature,
        top_p=gen.top_p,
        max_tokens=gen.max_new_tokens,
    )
    return rsp.choices[0].message.content.strip()


GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def generate_with_groq(prompt: str, model: str = "openai/gpt-oss-120b",
                       gen: GenerationConfig | None = None,
                       reasoning_effort: str = "low") -> str:
    """Generate with an open-weights model hosted on Groq.

    Groq exposes an OpenAI-compatible endpoint, so the same client library is
    reused with a different base URL. The attraction for this project is that
    the weights are open: the identical model can later be self-hosted inside a
    bank's own perimeter, which a proprietary hosted model cannot be.

    The gpt-oss family are reasoning models whose reasoning tokens are charged
    against `max_tokens`, so at a 200-token cap they return empty content
    unless the reasoning effort is turned down. `reasoning_effort='low'` keeps
    the budget available for the message. Models that do not accept the
    parameter are retried without it.

    Requires GROQ_API_KEY, which is free and needs no payment details.
    """
    import os

    from openai import OpenAI

    gen = gen or GenerationConfig()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=gen.temperature,
        top_p=gen.top_p,
        max_tokens=gen.max_new_tokens,
    )
    try:
        rsp = client.chat.completions.create(
            **kwargs, reasoning_effort=reasoning_effort
        )
    except Exception as exc:                                   # noqa: BLE001
        if "reasoning_effort" not in str(exc):
            raise
        rsp = client.chat.completions.create(**kwargs)
    text = (rsp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError(
            f"{model} returned empty content "
            f"(finish_reason={rsp.choices[0].finish_reason}); the reasoning "
            "budget may have consumed max_tokens"
        )
    return text


def list_groq_models() -> List[str]:
    """Model ids the current Groq key can call."""
    import os

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL)
    return sorted(m.id for m in client.models.list().data)


_GEMINI_CLIENT = None


def _gemini_client():
    """Cached Gen AI client; reads GEMINI_API_KEY (or GOOGLE_API_KEY)."""
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        import os

        from google import genai

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _GEMINI_CLIENT = genai.Client(api_key=key)
    return _GEMINI_CLIENT


def list_gemini_models() -> List[str]:
    """Model ids the current key can call, so the caller need not guess."""
    return sorted(
        m.name.removeprefix("models/")
        for m in _gemini_client().models.list()
        if "generateContent" in (getattr(m, "supported_actions", None) or
                                 getattr(m, "supported_generation_methods", None) or
                                 ["generateContent"])
    )


def generate_with_gemini(prompt: str, model: str = "gemini-3.1-flash-lite",
                         gen: GenerationConfig | None = None,
                         max_retries: int = 5,
                         thinking_budget: int = 0) -> str:
    """Generate via the Gemini Developer API.

    Two behaviours of the Gemini 3.x family are handled here.

    First, these are *thinking* models: tokens spent on internal reasoning are
    charged against `max_output_tokens`, so a budget sized for a short message
    leaves almost nothing for the message itself and the reply comes back
    truncated to a few words. `thinking_budget=0` disables reasoning for this
    task, which is a plain instruction-following job that does not need it.

    Second, the free tier is rate limited, so a 429 is an expected response
    rather than a fault; it is retried with exponential back-off.

    Requires GEMINI_API_KEY (or GOOGLE_API_KEY).
    """
    import time

    from google.genai import types

    gen = gen or GenerationConfig()
    kwargs = dict(
        temperature=gen.temperature,
        top_p=gen.top_p,
        max_output_tokens=gen.max_new_tokens,
    )
    if thinking_budget is not None:
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
        except Exception:                     # SDK too old for thinking_config
            pass
    cfg = types.GenerateContentConfig(**kwargs)
    delay = 8.0
    for attempt in range(1, max_retries + 1):
        try:
            rsp = _gemini_client().models.generate_content(
                model=model, contents=prompt, config=cfg
            )
            text = (rsp.text or "").strip()
            reason = getattr(
                (getattr(rsp, "candidates", None) or [None])[0], "finish_reason", None
            )
            if text and str(reason or "").upper().endswith(("STOP", "NONE")) or (
                    text and reason is None):
                return text
            if text and "MAX_TOKEN" in str(reason or "").upper():
                raise RuntimeError(
                    f"truncated at max_output_tokens={gen.max_new_tokens} "
                    f"(raise RETENTION_MAX_TOKENS or lower thinking_budget)"
                )
            if text:
                return text
            raise RuntimeError(f"empty response (finish_reason={reason})")
        except Exception as exc:                      # noqa: BLE001
            transient = any(s in str(exc).lower() for s in
                            ("429", "rate", "quota", "unavailable", "503", "500",
                             "deadline", "timeout"))
            if attempt == max_retries or not transient:
                raise
            logger.warning("Gemini attempt %d/%d failed (%s); retrying in %.0fs",
                           attempt, max_retries, type(exc).__name__, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Human-evaluation aggregation
# ---------------------------------------------------------------------------
def summarise_human_scores(df: pd.DataFrame, dims: Iterable[str] = ("Relevance", "Personalization",
                                                                    "Actionability", "Persuasiveness"),
                           kind_col: str = "kind") -> pd.DataFrame:
    """Mean +/- std per dimension, broken down by `kind_col` ('generic' vs 'llm')."""
    rows = []
    for kind, sub in df.groupby(kind_col):
        for d in dims:
            rows.append({
                "kind": kind, "dim": d,
                "mean": float(sub[d].mean()), "std": float(sub[d].std(ddof=1)),
                "n": int(sub[d].notna().sum()),
            })
    return pd.DataFrame(rows)


def krippendorff_alpha(ratings: np.ndarray) -> float:
    """Wraps the `krippendorff` package; ratings shape = (raters, items)."""
    import krippendorff

    return float(krippendorff.alpha(reliability_data=ratings, level_of_measurement="interval"))

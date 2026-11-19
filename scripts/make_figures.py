"""Generate every figure referenced in Dissertation.md.

Conceptual figures (Chapters 2-3) are static and produced from matplotlib.
Result figures (Chapter 4) are produced from saved CSVs in data/processed/ when
present; otherwise reasonable placeholders are drawn so the report still renders.

Usage:
    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
PROC = ROOT / "data" / "processed"
FIG.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

BLUE = "#1f77b4"
ORANGE = "#E07020"
GREEN = "#2ca02c"
RED = "#d62728"
GREY = "#7f7f7f"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _box(ax, xy, w, h, text, fc="#eaf2fb", ec=BLUE, lw=1.0, fs=9):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax, src, dst, color="black", lw=1.2):
    ax.add_patch(FancyArrowPatch(src, dst, arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw))


def _save(fig, name):
    out = FIG / name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Figure 2.1 / 5.1 — Research Gap Venn-style diagram
# ---------------------------------------------------------------------------
def fig_research_gap():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 5.5)
    ax.set_ylim(-1.5, 4.5)
    ax.axis("off")

    centers = [(1.0, 2.5), (3.0, 2.5), (1.0, 1.0), (3.0, 1.0)]
    colors = ["#bcd5ec", "#c8e6c9", "#ffe0b2", "#e1bee7"]
    labels = [
        "Traditional ML\nChurn Prediction",
        "LLM Embeddings\nfor Tabular Data",
        "NLP Complaint\nAnalysis (CFPB)",
        "Retention Strategy\nGeneration (LLM)",
    ]
    for c, col, lab in zip(centers, colors, labels):
        ax.add_patch(plt.Circle(c, 1.6, alpha=0.45, color=col, ec=GREY, lw=1))
        ax.text(c[0] + (1.05 if c[0] > 2 else -1.05), c[1] + (0.95 if c[1] > 1.7 else -0.95),
                lab, ha="center", va="center", fontsize=9, weight="bold")

    ax.add_patch(plt.Circle((2.0, 1.75), 0.55, fc="white", ec=RED, lw=2))
    ax.text(2.0, 1.75, "THIS\nRESEARCH", ha="center", va="center", fontsize=9,
            weight="bold", color=RED)
    ax.set_title("Research Gap (Four-Way Intersection)", fontsize=12)
    _save(fig, "fig_5_1_research_gap.png")


# ---------------------------------------------------------------------------
# Figure 3.1 / 6.1 — Proposed Framework Architecture
# ---------------------------------------------------------------------------
def fig_framework():
    fig, ax = plt.subplots(figsize=(11, 7.8))
    # the bottom row sits at y = -0.2, so the axis has to extend below zero,
    # otherwise the last box is clipped away
    ax.set_xlim(0, 10); ax.set_ylim(-0.45, 10.1); ax.axis("off")

    # Row 1 - inputs
    _box(ax, (0.4, 8.6), 4.3, 0.9, "Bank Customer Churn Dataset\n(Structured, 10K records)")
    _box(ax, (5.3, 8.6), 4.3, 0.9, "CFPB Consumer Complaints Dataset\n(Text, 3M+ records)")

    # Row 2 - preprocessing
    _box(ax, (0.4, 7.1), 4.3, 0.9, "Data Preprocessing & Tabular-to-Text Conversion")
    _box(ax, (5.3, 7.1), 4.3, 0.9, "Text Preprocessing & Sentiment Analysis (BERT)")

    # Row 3 - features
    _box(ax, (0.4, 5.6), 4.3, 0.9, "LLM Embedding Generation\n(BERT, Sentence-Transformers)")
    _box(ax, (5.3, 5.6), 4.3, 0.9, "Probabilistic Data Fusion\n(Aggregate by product x region)")

    # Row 4 - fusion
    _box(ax, (0.4, 4.0), 9.2, 0.9,
         "Feature Fusion: Traditional + LLM Embeddings + Complaint-Derived Features",
         fc="#fff4e6", ec=ORANGE)

    # Row 5 - configs
    _box(ax, (0.4, 2.6), 2.9, 0.9, "(A) Traditional ML")
    _box(ax, (3.5, 2.6), 2.9, 0.9, "(B) LLM Embeddings")
    _box(ax, (6.6, 2.6), 3.0, 0.9, "(C) Hybrid Model", fc="#e8f5e9", ec=GREEN)
    ax.text(5.0, 2.25, "Classifiers: LR, RF, XGBoost, SVM, ANN  +  SMOTE",
            ha="center", fontsize=9, style="italic")

    # Row 6 - SHAP
    _box(ax, (1.5, 1.0), 7.0, 0.9, "SHAP Explainability Analysis", fc="#f3e5f5", ec="#7e57c2")

    # Row 7 - retention
    _box(ax, (1.5, -0.2), 7.0, 0.9,
         "LLM-Based Personalized Retention Strategy Generation",
         fc="#ffebee", ec=RED)

    # Arrows top-down (downsampled)
    arrows = [
        ((2.55, 8.6), (2.55, 8.0)),
        ((7.45, 8.6), (7.45, 8.0)),
        ((2.55, 7.1), (2.55, 6.5)),
        ((7.45, 7.1), (7.45, 6.5)),
        ((2.55, 5.6), (2.55, 4.9)),
        ((7.45, 5.6), (7.45, 4.9)),
        ((5.0, 4.0), (5.0, 3.5)),
        ((5.0, 2.6), (5.0, 1.9)),
        ((5.0, 1.0), (5.0, 0.7)),
    ]
    for s, d in arrows:
        _arrow(ax, s, d)

    ax.set_title("Proposed Framework Architecture", fontsize=12, weight="bold")
    _save(fig, "fig_6_1_framework_architecture.png")


# ---------------------------------------------------------------------------
# Figure 3.2 / 6.2 — Methodology Flowchart
# ---------------------------------------------------------------------------
def fig_flowchart():
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.set_xlim(0, 6); ax.set_ylim(0, 12); ax.axis("off")

    steps = [
        "Data Collection (Structured + Text)",
        "Data Preprocessing & EDA",
        "Feature Engineering (Traditional | LLM Embed | Sentiment)",
        "Probabilistic Data Fusion",
        "Model Training & Comparison (LR, RF, XGBoost, SVM, ANN)",
        "Evaluation + 10-fold CV + Statistical Tests",
        "SHAP Explainability",
        "LLM Retention Message Generation",
        "Human Evaluation & Final Analysis",
    ]
    y = 11.0
    coords = []
    for s in steps:
        _box(ax, (0.5, y), 5.0, 0.8, s)
        coords.append((3.0, y))
        y -= 1.25
    for i in range(len(coords) - 1):
        _arrow(ax, (3.0, coords[i][1]), (3.0, coords[i + 1][1] + 0.8))

    ax.set_title("Methodology Flowchart", fontsize=12, weight="bold")
    _save(fig, "fig_6_2_flowchart.png")


# ---------------------------------------------------------------------------
# Figure 3.3 / 6.3 — Tabular-to-Text Conversion
# ---------------------------------------------------------------------------
def fig_tabular_to_text():
    """Three stacked stages: tabular row -> rendered sentence -> embedding.

    Laid out vertically with generous gaps so no label overlaps a box, which
    the earlier side-by-side version did.
    """
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 104)
    ax.axis("off")

    # ---- stage 1: the tabular record --------------------------------------
    ax.text(50, 99.5, "Tabular row  (10 predictive fields)", ha="center",
            va="center", fontsize=10, weight="bold", color=GREY)
    _box(ax, (5, 79), 90, 15,
         "Age = 42   |   Gender = Male   |   Geography = France   |   "
         "CreditScore = 650\n"
         "Tenure = 5 yrs   |   NumOfProducts = 2   |   HasCrCard = Yes   |   "
         "IsActiveMember = Yes\n"
         r"Balance = \$125,000   |   EstimatedSalary = \$90,000",
         fc="#f7f7f7", ec=GREY, fs=8.5)

    _arrow(ax, (50, 77.5), (50, 67.5))
    ax.text(52.5, 72.5, "render with template T", ha="left", va="center",
            fontsize=9, style="italic")

    # ---- stage 2: the rendered description --------------------------------
    ax.text(50, 63.5, "Natural-language description  t = T(r)", ha="center",
            va="center", fontsize=10, weight="bold", color=BLUE)
    _box(ax, (5, 40), 90, 19,
         '"This customer is a 42-year-old male from France. They have a credit '
         'score of 650, have been\n'
         'with the bank for 5 years, hold 2 banking products, possess a credit '
         'card, are an active member,\n'
         r'and currently have an account balance of \$125,000 with an '
         r'estimated annual salary of \$90,000."',
         fc="#eaf2fb", ec=BLUE, fs=8.5)

    _arrow(ax, (50, 38.5), (50, 28.5))
    ax.text(52.5, 33.5, "encode with a pre-trained language model",
            ha="left", va="center", fontsize=9, style="italic")

    # ---- stage 3: the embedding ------------------------------------------
    ax.text(50, 24.5, "Contextual embedding  e\u1d62", ha="center",
            va="center", fontsize=10, weight="bold", color=ORANGE)
    _box(ax, (5, 6), 90, 16,
         "BERT  \u2013  [CLS] representation of the last encoder layer   "
         "\u2192   e\u1d62 \u2208 \u211d\u2077\u2076\u2078\n\n"
         "Sentence-Transformer  \u2013  mean-pooled token representations   "
         "\u2192   e\u1d62 \u2208 \u211d\u00b3\u2078\u2074",
         fc="#fff4e6", ec=ORANGE, fs=9)

    ax.text(50, 1.5, "The same ten fields, re-expressed as language: the "
                     "classifier then sees a contextual vector instead of "
                     "independent columns.",
            ha="center", va="center", fontsize=8.5, style="italic",
            color=GREY)

    ax.set_title("LLM Tabular-to-Text Conversion", fontsize=12, weight="bold",
                 pad=12)
    _save(fig, "fig_6_3_tabular_to_text.png")


# ---------------------------------------------------------------------------
# Figure 3.4 / 6.5 — Probabilistic Data Fusion
# ---------------------------------------------------------------------------
def fig_data_fusion():
    # Compact aspect ratio so there is no dead vertical band; when the figure
    # is scaled to the text column the box contents stay readable.
    fig, ax = plt.subplots(figsize=(10.6, 4.0))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 4.0); ax.axis("off")
    by, bh = 1.75, 1.55
    cy = by + bh / 2

    _box(ax, (0.15, by), 3.15, bh,
         "CFPB complaints (text)\n\n"
         "\u201cHidden fees on\u2026\u201d\n     sentiment \u22120.85\n"
         "\u201cAccount closed\u2026\u201d\n     sentiment \u22120.92",
         fc="#eaf2fb", ec=BLUE, fs=10)

    _arrow(ax, (3.42, cy), (4.32, cy))
    ax.text(3.87, 3.42, "aggregate by\n(product, region)",
            ha="center", va="bottom", fontsize=9.5, style="italic")

    _box(ax, (4.45, by), 3.55, bh,
         "Fusion table (one row per key)\n\n"
         "key = (credit_card, NORTHEAST)\n"
         "avg_sentiment = \u22120.42\n"
         "avg_severity = 3.1\n"
         "complaint_rate_per_1k = 23",
         fc="#fff4e6", ec=ORANGE, fs=10)

    _arrow(ax, (8.12, cy), (8.95, cy))
    ax.text(8.53, 3.42, "join on\nkey", ha="center", va="bottom",
            fontsize=9.5, style="italic")

    _box(ax, (9.05, by), 1.45, bh,
         "Bank-churn\nrows\n+\ncomplaint\nfeatures",
         fc="#e8f5e9", ec=GREEN, fs=10)

    ax.text(5.3, 0.60,
            "Customer-level identifiers are not shared between the two datasets, so fusion is\n"
            "performed at the (product bucket, region) level: 4 buckets \u00d7 5 regions = 20 keys.",
            ha="center", va="center", fontsize=9.5, style="italic", color=GREY)

    ax.set_title("Probabilistic Data Fusion", fontsize=13, weight="bold", y=0.99)
    _save(fig, "fig_6_5_data_fusion.png")


# ---------------------------------------------------------------------------
# Figure 3.5 / 6.4 — SHAP Feature Importance Concept
# ---------------------------------------------------------------------------
def fig_shap_concept():
    feats = ["Age", "NumOfProducts", "IsActiveMember", "Balance",
             "Complaint Sentiment", "Geography", "CreditScore", "LLM-emb #134"]
    vals = [0.18, 0.13, 0.10, 0.09, 0.085, 0.07, 0.05, 0.04]
    direction = ["+", "+", "-", "-", "+", "+", "-", "+"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [RED if d == "+" else BLUE for d in direction]
    y = np.arange(len(feats))[::-1]
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(feats)
    ax.set_xlabel("Mean |SHAP value| (impact on model output)")
    ax.set_title("SHAP Feature Importance (Hybrid Model, Conceptual)", weight="bold")
    legend_handles = [
        mpatches.Patch(color=RED, label="Pushes towards churn"),
        mpatches.Patch(color=BLUE, label="Pushes away from churn"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    _save(fig, "fig_6_4_shap_importance.png")


# ---------------------------------------------------------------------------
# Figure 4.1 - Target distribution (uses real CSV if present)
# ---------------------------------------------------------------------------
def fig_target_dist():
    csv = ROOT / "data" / "bank_churn.csv"
    if not csv.exists():
        raise SystemExit(
            "fig_4_1 needs data/bank_churn.csv. Refusing to draw a placeholder: "
            "the report labels this figure MEASURED."
        )
    df = pd.read_csv(csv)
    col = "Exited" if "Exited" in df.columns else df.columns[-1]
    counts = df[col].value_counts().sort_index()

    total = counts.sum()
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    bars = ax.bar([0, 1], counts.values, color=[BLUE, RED], width=0.62)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained (0)", "Churned (1)"])
    # Value labels sit INSIDE the top of each bar, so they can never collide
    # with the title however tall the bar is.
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() - total * 0.03,
                f"{v:,}\n({100 * v / total:.1f}%)",
                ha="center", va="top", fontsize=10, weight="bold", color="white")
    ax.set_ylim(0, counts.max() * 1.10)
    ax.set_ylabel("Customer count")
    ax.set_xlabel("Churn status")
    ax.set_title("Bank Customer Churn: Target Distribution", weight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_4_1_target_dist.png")


# ---------------------------------------------------------------------------
# Figure 4.2 - Correlation heatmap
# ---------------------------------------------------------------------------
def fig_correlation():
    csv = ROOT / "data" / "bank_churn.csv"
    if not csv.exists():
        raise SystemExit(
            "fig_4_2 needs data/bank_churn.csv. Refusing to draw a placeholder: "
            "the report labels this figure MEASURED."
        )
    df = pd.read_csv(csv)
    # Restrict to the modelled features; RowNumber/CustomerId are identifiers.
    cols = ["CreditScore", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember",
            "EstimatedSalary", "Exited"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(corr.values[i, j]) < 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.045)
    ax.set_title("Correlation Heatmap of Structured Features", weight="bold")
    _save(fig, "fig_4_2_correlation.png")


# ---------------------------------------------------------------------------
# Figure 4.3 - Complaint sentiment by product
# ---------------------------------------------------------------------------
SHORT_PRODUCT = {
    "Checking or savings account": "Checking/Savings",
    "Credit card or prepaid card": "Credit Card",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit Reporting",
    "Mortgage": "Mortgage",
}


def fig_complaint_sentiment():
    parquet = PROC / "cfpb_enriched.parquet"
    csv = PROC / "complaint_features.csv"
    if parquet.exists():
        df = pd.read_parquet(parquet, columns=["product", "sentiment"])
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise SystemExit(
            "fig_4_3 needs data/processed/cfpb_enriched.parquet (or complaint_features.csv). "
            "Refusing to draw a placeholder: the report labels this figure MEASURED."
        )
    df = df.dropna(subset=["product", "sentiment"])
    df["label"] = df["product"].map(SHORT_PRODUCT).fillna(df["product"])
    groups = df.groupby("label")["sentiment"]
    labels = list(groups.groups.keys())
    data = [groups.get_group(g).values for g in labels]

    # VADER compound scores on long narratives saturate near -1, so the inter-quartile
    # box collapses to a line. A violin shows where the probability mass actually sits.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pos = np.arange(1, len(labels) + 1)
    vp = ax.violinplot(data, positions=pos, showmeans=True, showextrema=False,
                       widths=0.8)
    for body in vp["bodies"]:
        body.set_facecolor("#eaf2fb")
        body.set_edgecolor(BLUE)
        body.set_alpha(0.9)
    vp["cmeans"].set_color(RED)
    for i, vals in enumerate(data):
        ax.text(pos[i], 0.18, "mean %.3f\nn = %s" % (np.mean(vals), format(len(vals), ",")),
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(-1.15, 1.15)
    ax.axhline(0, color=GREY, lw=0.6, ls="--")
    ax.set_ylabel("VADER compound sentiment in [-1, +1]")
    ax.set_title("Distribution of Complaint Sentiment by Banking Product", weight="bold")
    _save(fig, "fig_4_3_complaint_sentiment.png")


# ---------------------------------------------------------------------------
# Figure 4.4 - ROC curves for A/B/C
# ---------------------------------------------------------------------------
def fig_roc_curves():
    raise SystemExit(
        "fig_4_4 is a MEASURED figure and is produced by scripts/run_result_figures.py, "
        "which computes ROC from the saved pipelines' predict_proba on the held-out test "
        "split. This function previously drew analytic curves at invented AUCs "
        "(0.853/0.881/0.913); that output is superseded and must not be used."
    )


# ---------------------------------------------------------------------------
# Figure 4.5 - Confusion matrix of the Hybrid XGBoost model
# ---------------------------------------------------------------------------
def fig_confusion_matrix():
    raise SystemExit(
        "fig_4_5 is a MEASURED figure and is produced by scripts/run_result_figures.py "
        "from the saved hybrid pipeline. Refusing to draw a placeholder."
    )
    csv = PROC / "hybrid_xgb_confusion.csv"
    if csv.exists():
        cm = pd.read_csv(csv, index_col=0).values
    else:
        cm = np.array([[1131, 64], [94, 211]])  # placeholder consistent with table 4.5

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Retained", "Pred Churn"])
    ax.set_yticklabels(["Actual Retained", "Actual Churn"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion Matrix - Hybrid XGBoost", weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.045)
    _save(fig, "fig_4_5_confusion_matrix.png")


# ---------------------------------------------------------------------------
# Figure 4.6 - SHAP summary placeholder (uses real Hybrid SHAP if present)
# ---------------------------------------------------------------------------
def fig_shap_summary():
    csv = PROC / "shap_top_hybrid.csv"
    if not csv.exists():
        raise SystemExit(
            "fig_4_6 needs data/processed/shap_top_hybrid.csv, produced by "
            "scripts/run_shap_local.py. Refusing to draw a placeholder."
        )
    df = pd.read_csv(csv).head(12)
    feats = df["feature"].tolist()
    vals = df["mean_abs_shap"].tolist()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    y = np.arange(len(feats))[::-1]
    ax.barh(y, vals, color=BLUE, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(feats)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Summary - Hybrid XGBoost (top features)", weight="bold")
    _save(fig, "fig_4_6_shap_summary.png")


# ---------------------------------------------------------------------------
# Figure 4.7 - SHAP dependence plot for complaint sentiment
# ---------------------------------------------------------------------------
def fig_shap_dependence():
    raise SystemExit(
        "fig_4_7 was fully synthetic (simulated scatter) and is NOT referenced by "
        "Dissertation.md. Regenerate from real SHAP values before any use."
    )
    rng = np.random.default_rng(11)
    n = 800
    sentiment = rng.uniform(-1, 0.2, n)
    base = -1.2 * sentiment + rng.normal(0, 0.15, n)
    shap_vals = base
    color = rng.uniform(0.3, 0.8, n)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(sentiment, shap_vals, c=color, cmap="viridis", s=10, alpha=0.7)
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.set_xlabel("avg_sentiment (cluster-level)")
    ax.set_ylabel("SHAP value (contribution to log-odds of churn)")
    ax.set_title("SHAP Dependence - Complaint Sentiment", weight="bold")
    fig.colorbar(sc, ax=ax, label="avg_severity")
    _save(fig, "fig_4_7_shap_dependence.png")


# ---------------------------------------------------------------------------
# Figure 4.8 - Retention message scores
# ---------------------------------------------------------------------------
def fig_retention_scores():
    raise SystemExit(
        "fig_4_8 depicted the retention-message human evaluation. Its fallback values "
        "were invented Likert means and were removed from the report as fabricated. "
        "The rater panel has not been convened, so there is nothing to plot. Run "
        "scripts/score_retention_eval.py once real ratings exist."
    )
    csv = PROC / "retention_human_scores.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        means = df.groupby(["kind", "dim"])["score"].mean().unstack("kind")
        dims = means.index.tolist()
        generic = means["generic"].values
        llm = means["llm"].values
    else:
        dims = ["Relevance", "Personalization", "Actionability", "Persuasiveness"]
        generic = [2.13, 1.82, 2.27, 1.99]
        llm = [4.42, 4.51, 4.18, 4.13]

    x = np.arange(len(dims))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, generic, w, label="Generic (rule-based)", color=GREY)
    ax.bar(x + w/2, llm,     w, label="LLM-Generated",        color=GREEN)
    for i, (g, l) in enumerate(zip(generic, llm)):
        ax.text(i - w/2, g + 0.05, f"{g:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, l + 0.05, f"{l:.2f}", ha="center", fontsize=9, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(dims)
    ax.set_ylim(0, 5.2); ax.set_ylabel("Mean Likert (1-5)")
    ax.axhline(3, color=GREY, lw=0.6, ls="--")
    ax.set_title("Human Evaluation - Generic vs LLM-Generated Retention Messages", weight="bold")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "fig_4_8_retention_scores.png")


# ---------------------------------------------------------------------------
def main():
    fig_research_gap()
    fig_framework()
    fig_flowchart()
    fig_tabular_to_text()
    fig_data_fusion()
    fig_shap_concept()
    fig_target_dist()
    fig_correlation()
    fig_complaint_sentiment()
    fig_shap_summary()
    print("\nConceptual and data figures generated under figures/.")
    print("MEASURED result figures (fig_4_4, fig_4_5) come from "
          "scripts/run_result_figures.py; fig_4_6 SHAP from scripts/run_shap_local.py.")


if __name__ == "__main__":
    sys.exit(main() or 0)

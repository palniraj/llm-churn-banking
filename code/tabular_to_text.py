"""Tabular-to-text rendering for the LLM-embedding pathway."""
from __future__ import annotations

import pandas as pd


def declarative(row: pd.Series) -> str:
    """Default declarative template (used for headline numbers)."""
    return (
        f"This customer is a {int(row['Age'])}-year-old {str(row['Gender']).lower()} "
        f"from {row['Geography']}. They have a credit score of {int(row['CreditScore'])}, "
        f"have been with the bank for {int(row['Tenure'])} years, hold "
        f"{int(row['NumOfProducts'])} banking products, "
        f"{'possess' if int(row['HasCrCard']) else 'do not possess'} a credit card, "
        f"are {'an active' if int(row['IsActiveMember']) else 'an inactive'} member, "
        f"and currently have an account balance of ${float(row['Balance']):,.0f} "
        f"with an estimated annual salary of ${float(row['EstimatedSalary']):,.0f}."
    )


def qa_style(row: pd.Series) -> str:
    """Q&A style template (ablation variant)."""
    return (
        f"Q: Who is this customer? A: A {int(row['Age'])}-year-old "
        f"{str(row['Gender']).lower()} from {row['Geography']}. "
        f"Q: What is the credit score? A: {int(row['CreditScore'])}. "
        f"Q: Tenure? A: {int(row['Tenure'])} years. "
        f"Q: Products? A: {int(row['NumOfProducts'])}. "
        f"Q: Has credit card? A: {'Yes' if int(row['HasCrCard']) else 'No'}. "
        f"Q: Active member? A: {'Yes' if int(row['IsActiveMember']) else 'No'}. "
        f"Q: Balance? A: ${float(row['Balance']):,.0f}. "
        f"Q: Estimated salary? A: ${float(row['EstimatedSalary']):,.0f}."
    )


def compact(row: pd.Series) -> str:
    """Compact key=value template (ablation variant)."""
    return (
        f"Age={int(row['Age'])}; Gender={row['Gender']}; Geography={row['Geography']}; "
        f"CreditScore={int(row['CreditScore'])}; Tenure={int(row['Tenure'])}; "
        f"Products={int(row['NumOfProducts'])}; "
        f"HasCard={'Yes' if int(row['HasCrCard']) else 'No'}; "
        f"ActiveMember={'Yes' if int(row['IsActiveMember']) else 'No'}; "
        f"Balance={float(row['Balance']):.0f}; Salary={float(row['EstimatedSalary']):.0f}."
    )


TEMPLATES = {
    "declarative": declarative,
    "qa": qa_style,
    "compact": compact,
}


def render(df: pd.DataFrame, template: str = "declarative") -> pd.Series:
    """Vectorise the chosen template across `df` rows."""
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template '{template}'. Pick one of {list(TEMPLATES)}.")
    fn = TEMPLATES[template]
    return df.apply(fn, axis=1)

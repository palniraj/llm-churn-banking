"""Stage 3: build the blinded rating sheets that the human panel fills in.

Each rater receives one CSV holding all 50 profiles twice: once with the
generic message and once with the generated message. Within every profile the
two messages are shuffled and labelled only "A" and "B", the profile order is
shuffled per rater, and nothing in the sheet reveals which system produced
which message. The mapping is written to a separate key file that raters must
not see.

Input :  data/processed/retention_messages.csv        (stage 2)
Output:  data/processed/rating_sheets/rater_<k>.csv   one per rater
         data/processed/rating_sheets/README.txt      rater instructions
         data/processed/rating_key.csv                blinding key, keep aside
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from code import config

DIMENSIONS = ["Relevance", "Personalization", "Actionability", "Persuasiveness"]

INSTRUCTIONS = """RETENTION MESSAGE RATING TASK
=============================

You will see {n} bank customer profiles. Each profile is followed by two
candidate retention messages, labelled A and B. They come from two different
systems, but the labels are randomised for every profile, so A is not
consistently one system.

For EACH message, enter a whole number from 1 to 5 in the four score columns:

  Relevance        1 = unrelated to this customer
                   5 = directly addresses their likely reason for leaving
  Personalization  1 = generic, could be sent to anyone
                   5 = clearly tailored to this specific customer
  Actionability    1 = no clear next step
                   5 = a clear, implementable next step
  Persuasiveness   1 = would not convince this customer to stay
                   5 = would very likely convince them to stay

Rules
-----
- Score every row. Leave no blanks.
- Judge only the message text, not the customer's likelihood of churning.
- Do not go back and revise earlier rows to make them consistent.
- Work through the sheet in the order given; it is randomised deliberately.
- Do not discuss your scores with the other raters until all sheets are in.

Save the file with the same name when you are done. Do not add, remove or
reorder rows or columns.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", type=int, default=4,
                    help="number of rating sheets to produce (3 to 5 per the protocol)")
    ap.add_argument("--seed", type=int, default=config.PRIMARY_SEED)
    args = ap.parse_args()

    src = config.PROCESSED_DIR / "retention_messages.csv"
    if not src.exists():
        raise SystemExit(f"{src} missing. Run scripts/run_retention_generate.py first.")
    df = pd.read_csv(src).fillna({"message_llm": "", "message_generic": ""})

    blank = df["message_llm"].str.strip() == ""
    if blank.any():
        raise SystemExit(
            f"{int(blank.sum())} of {len(df)} profiles have no generated message. "
            "Run stage 2 with a real backend before building rating sheets."
        )

    outdir = config.PROCESSED_DIR / "rating_sheets"
    outdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    key_rows = []
    for k in range(1, args.raters + 1):
        rows = []
        order = rng.permutation(len(df))          # per-rater profile order
        for pos, ridx in enumerate(order, start=1):
            r = df.iloc[ridx]
            pair = [("generic", r["message_generic"]), ("llm", r["message_llm"])]
            if rng.random() < 0.5:                # per-profile A/B flip
                pair = pair[::-1]
            for label, (kind, text) in zip("AB", pair):
                rows.append({
                    "sheet_position": pos,
                    "profile_id": r["profile_id"],
                    "customer_profile": r["profile"],
                    "message_label": label,
                    "message_text": text,
                    **{d: "" for d in DIMENSIONS},
                })
                key_rows.append({
                    "rater": k, "profile_id": r["profile_id"],
                    "message_label": label, "kind": kind,
                })
        sheet = pd.DataFrame(rows)
        path = outdir / f"rater_{k}.csv"
        sheet.to_csv(path, index=False)
        print(f"  wrote {path}  ({len(sheet)} rows to score)")

    (outdir / "README.txt").write_text(INSTRUCTIONS.format(n=len(df)), encoding="utf-8")
    pd.DataFrame(key_rows).to_csv(config.PROCESSED_DIR / "rating_key.csv", index=False)

    print(f"\n{args.raters} sheets in {outdir}")
    print(f"Blinding key: {config.PROCESSED_DIR / 'rating_key.csv'}  (do not share with raters)")
    print(f"Each rater scores {len(df) * 2} rows across {len(DIMENSIONS)} dimensions.")
    print("\nNext: collect the completed sheets, then run scripts/score_retention_eval.py")


if __name__ == "__main__":
    main()

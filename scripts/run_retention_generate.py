"""Stage 2: generate the personalized retention message for each profile.

This is the only stage that needs a generative model. Three backends are
supported; pick whichever is available.

    # OpenAI-compatible API (fastest, needs OPENAI_API_KEY)
    python scripts/run_retention_generate.py --backend openai --model gpt-4o-mini

    # Local or Colab GPU via transformers (4-bit Llama 3 8B by default)
    python scripts/run_retention_generate.py --backend hf

    # Dry run: no model, writes an empty message column so stages 3 and 4 can
    # be rehearsed end to end before the real generation
    python scripts/run_retention_generate.py --backend none

Input :  data/processed/retention_profiles.csv   (stage 1)
Output:  data/processed/retention_messages.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from code import config, retention_llm


def load_dotenv(path: Path) -> bool:
    """Read KEY=value lines from a .env file into the environment.

    `.env` is listed in .gitignore, so a key placed there is never committed.
    Never put an API key in README.md, a notebook, or any tracked file.
    """
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend",
                    choices=("groq", "gemini", "openai", "hf", "none"),
                    default="none")
    ap.add_argument("--model", default=None,
                    help="model id; defaults per backend")
    ap.add_argument("--limit", type=int, default=None,
                    help="generate for the first N profiles only (smoke test)")
    ap.add_argument("--sleep", type=float, default=None,
                    help="seconds between calls; default 7 for gemini to stay "
                         "inside the free tier's ~10 requests per minute")
    ap.add_argument("--force-model-change", action="store_true",
                    help="discard existing messages and regenerate all of them "
                         "with a different model")
    ap.add_argument("--thinking-budget", type=int, default=0,
                    help="Gemini 3.x reasoning-token budget. Reasoning tokens "
                         "count against max_output_tokens, so the default 0 "
                         "disables it and keeps the whole budget for the "
                         "message; without this the reply truncates to a few "
                         "words")
    ap.add_argument("--list-models", action="store_true",
                    help="print the model ids this key can call, then exit")
    args = ap.parse_args()

    if load_dotenv(ROOT / ".env"):
        print("loaded credentials from .env (untracked)")

    NEEDS_KEY = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
                 "groq": "GROQ_API_KEY"}
    var = NEEDS_KEY.get(args.backend)
    if var and not os.environ.get(var) and not (
            args.backend == "gemini" and os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit(
            f"{var} is not set.\n"
            "  Safest, leaves no trace in shell history or on screen:\n"
            f"      read -rsp 'key: ' {var} && export {var}\n"
            "  Or put one line in thesis/.env (already gitignored):\n"
            f"      {var}=...\n"
            "  Do not paste a key into README.md or any tracked file.\n"
            + {"gemini": "  A Gemini key is free at https://aistudio.google.com/apikey\n",
               "groq": "  A Groq key is free at https://console.groq.com/keys\n"}.get(args.backend, "")
        )

    if args.list_models:
        if args.backend == "gemini":
            names = retention_llm.list_gemini_models()
        elif args.backend == "groq":
            names = retention_llm.list_groq_models()
        else:
            raise SystemExit("--list-models supports --backend gemini or groq")
        for name in names:
            print(" ", name)
        return

    profiles = config.PROCESSED_DIR / "retention_profiles.csv"
    partial = config.PROCESSED_DIR / "retention_messages.csv"
    if not profiles.exists():
        raise SystemExit(f"{profiles} missing. Run scripts/run_retention_profiles.py first.")
    # Continue from a part-finished run when one exists, otherwise start from
    # the profiles. Reading the profiles unconditionally would silently discard
    # everything generated so far.
    src = partial if partial.exists() else profiles
    print(f"reading {src.name}")
    df = pd.read_csv(src)
    # An all-empty column round-trips through CSV as float64 NaN, and pandas 3
    # refuses to assign a string into a float column, so force text dtype up
    # front for the two columns this script writes.
    for col in ("message_llm", "generation_model"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype("object")

    # Resume support. Free-tier daily caps are per model (gemini-3.6-flash
    # allows only 20 requests/day), so a run may legitimately stop part way and
    # be continued tomorrow. Already-generated rows are skipped.
    have = df["message_llm"].str.strip() != ""
    used = sorted(set(df.loc[have, "generation_model"]) - {""})
    if have.any():
        print(f"{have.sum()} of {len(df)} messages already generated"
              + (f" with {', '.join(used)}" if used else ""))

    # All rated messages must come from one model, or the comparison is not a
    # comparison. Refuse a silent mid-run model change.
    if have.any() and used and args.backend != "none":
        target = args.model or DEFAULT_MODEL[args.backend]
        if target not in used:
            if not args.force_model_change:
                raise SystemExit(
                    f"Existing messages were generated with {', '.join(used)}, "
                    f"but you asked for {target}.\n"
                    "Mixing models across profiles would invalidate the human "
                    "comparison. Either continue with the original model, or "
                    "pass --force-model-change to discard the existing "
                    f"{have.sum()} messages and start over with {target}."
                )
            print(f"discarding {have.sum()} messages generated with "
                  f"{', '.join(used)}; regenerating all with {target}")
            df["message_llm"] = ""
            df["generation_model"] = ""
            have = df["message_llm"].str.strip() != ""

    pending = df[~have]
    todo = pending if args.limit is None else pending.head(args.limit)
    if todo.empty:
        print("Nothing to do: every profile already has a message.")
        print("\nNext: stage 3, scripts/make_rating_sheets.py")
        return
    print(f"{len(todo)} profiles to generate with backend '{args.backend}'")

    # gemini-2.5-* now 404s on current keys; 3.6-flash is fast, stable (not a
    # -preview id) and grounds its output in the supplied profile.
    DEFAULT_MODEL = {"groq": "openai/gpt-oss-120b",
                     "gemini": "gemini-3.6-flash", "openai": "gpt-4o-mini",
                     "hf": "meta-llama/Meta-Llama-3-8B-Instruct"}
    # Groq free tier allows ~30 requests/minute, so 2.5s pacing is ample.
    DEFAULT_SLEEP = {"groq": 2.5, "gemini": 7.0, "openai": 0.0, "hf": 0.0}

    if args.backend == "none":
        print("Dry run: leaving message_llm empty.")
        model = ""
    else:
        model = args.model or DEFAULT_MODEL[args.backend]
        pause = args.sleep if args.sleep is not None else DEFAULT_SLEEP[args.backend]
        gen = retention_llm.GenerationConfig()
        print(f"model: {model}" + (f"   pacing: {pause:.0f}s between calls"
                                   if pause else ""))
        started = time.time()
        for n, (i, row) in enumerate(todo.iterrows(), start=1):
            if args.backend == "groq":
                text = retention_llm.generate_with_groq(row["prompt"], model, gen)
            elif args.backend == "gemini":
                text = retention_llm.generate_with_gemini(
                    row["prompt"], model, gen,
                    thinking_budget=args.thinking_budget)
            elif args.backend == "openai":
                text = retention_llm.generate_with_openai(row["prompt"], model, gen)
            else:
                text = retention_llm.generate_with_hf(row["prompt"], model, gen)
            df.loc[i, "message_llm"] = text
            df.loc[i, "generation_model"] = model
            print(f"  [{n}/{len(todo)}] {row['profile_id']}  {len(text.split())} words"
                  f"  ({(time.time() - started) / n:.1f}s/msg)")
            # save as we go: a rate-limit stop midway loses nothing
            df.to_csv(config.PROCESSED_DIR / "retention_messages.csv", index=False)
            if pause and n < len(todo):
                time.sleep(pause)
        print(f"\nGenerated {len(todo)} messages in {time.time() - started:.0f}s")

    out = config.PROCESSED_DIR / "retention_messages.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    missing = int((df["message_llm"].fillna("").str.strip() == "").sum())
    if missing:
        print(f"  ! {missing} profiles still have no generated message")
    print("\nNext: stage 3, scripts/make_rating_sheets.py")


if __name__ == "__main__":
    main()

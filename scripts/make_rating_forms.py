"""Stage 3b: turn each blinded rating sheet into a self-contained HTML form.

Asking a rater to type numbers into a 100-row CSV whose cells hold paragraphs of
text invites mistakes and abandonment. This produces one standalone .html file
per rater instead: it opens in any browser with no install and no account, keeps
partial progress in the browser so a rater can stop and come back, refuses to
export until every item is scored, and downloads a CSV in exactly the shape
scripts/score_retention_eval.py expects.

Input :  data/processed/rating_sheets/rater_<k>.csv   (stage 3)
Output:  data/processed/rating_sheets/rater_<k>.html  (send these out)
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from code import config

DIMENSIONS = ["Relevance", "Personalization", "Actionability", "Persuasiveness"]
ANCHORS = {
    "Relevance": ("unrelated to this customer",
                  "directly addresses their likely reason for leaving"),
    "Personalization": ("generic, could be sent to anyone",
                        "clearly tailored to this specific customer"),
    "Actionability": ("no clear next step",
                      "a clear, implementable next step"),
    "Persuasiveness": ("would not convince this customer to stay",
                       "would very likely convince them to stay"),
}

TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retention message rating &mdash; rater __K__</title>
<style>
 body { font: 16px/1.55 system-ui, sans-serif; max-width: 820px; margin: 0 auto;
        padding: 24px 18px 120px; color: #1a1a1a; }
 h1 { font-size: 22px; margin: 0 0 4px; }
 .intro { background: #f4f7fb; border: 1px solid #d7e2ef; border-radius: 8px;
           padding: 14px 18px; margin: 16px 0 28px; }
 .intro ul { margin: 8px 0 0 18px; padding: 0; }
 .item { border: 1px solid #dcdcdc; border-radius: 8px; padding: 16px 18px;
          margin: 0 0 20px; }
 .item.done { border-color: #bcd9bc; background: #fbfefb; }
 .pos { color: #777; font-size: 13px; letter-spacing: .04em; text-transform: uppercase; }
 .profile { background: #f7f7f7; border-left: 3px solid #bbb; padding: 8px 12px;
             margin: 8px 0 14px; font-size: 14.5px; color: #333; }
 .msg { white-space: pre-wrap; background: #fff; border: 1px solid #eee;
         border-radius: 6px; padding: 12px 14px; margin-bottom: 16px; }
 table.dims { border-collapse: collapse; width: 100%; }
 table.dims td { padding: 7px 4px; border-bottom: 1px solid #f0f0f0;
                  vertical-align: middle; }
 td.name { width: 30%; font-weight: 600; font-size: 14.5px; }
 td.anchor { width: 30%; font-size: 12.5px; color: #666; }
 .scale label { display: inline-block; margin-right: 9px; cursor: pointer;
                 font-size: 15px; }
 .bar { position: fixed; left: 0; right: 0; bottom: 0; background: #fff;
         border-top: 1px solid #ccc; padding: 11px 18px; display: flex;
         gap: 16px; align-items: center; justify-content: space-between; }
 button { font-size: 15px; padding: 9px 18px; border-radius: 6px;
           border: 1px solid #1f4e79; background: #1f4e79; color: #fff;
           cursor: pointer; }
 button[disabled] { background: #bbb; border-color: #bbb; cursor: not-allowed; }
 #status { font-size: 14.5px; }
</style></head><body>
<h1>Retention message rating</h1>
<p class="pos">Rater __K__ &middot; __N__ messages to score</p>

<div class="intro">
 <strong>What to do.</strong> Each block shows one bank customer, then one candidate
 retention message. Score the message on the four dimensions using 1 to 5.
 <ul>
  <li>The messages come from two different systems. Which is which is hidden, and
      the order is shuffled, so please judge each on its own merits.</li>
  <li>Judge only the <em>message</em>, not whether the customer seems likely to leave.</li>
  <li>Do not go back to make earlier scores consistent. First reaction is what we want.</li>
  <li>Your progress is saved in this browser, so you can close the page and return.</li>
  <li>When all items are scored, the download button at the bottom activates.
      Send the downloaded file back.</li>
 </ul>
 <strong>Please do not discuss your scores with the other raters until all sheets are in.</strong>
 <p style="margin:12px 0 0;font-size:13.5px;color:#555">
  <strong>About your participation.</strong> This rating task supports an M.Sc.
  dissertation on customer-churn prediction. It takes roughly 45 to 60 minutes.
  The customer profiles are synthetic records from a public dataset and describe
  no real person. Only your numeric scores are collected, they are reported only
  as group averages, and you are not named in the dissertation. Participation is
  voluntary and you may stop at any point; if you stop, simply do not send the
  file back.
 </p>
</div>

<div id="items"></div>

<div class="bar">
  <span id="status">0 of __N__ scored</span>
  <button id="dl" disabled>Download my scores</button>
</div>

<script>
const ROWS = __ROWS__;
const DIMS = __DIMS__;
const ANCHORS = __ANCHORS__;
const KEY = "retention_rater___K__";
const saved = JSON.parse(localStorage.getItem(KEY) || "{}");

const wrap = document.getElementById("items");
ROWS.forEach((r, idx) => {
  const div = document.createElement("div");
  div.className = "item"; div.id = "item" + idx;
  let rows = "";
  DIMS.forEach(d => {
    let scale = "";
    for (let v = 1; v <= 5; v++) {
      const id = `q${idx}_${d}_${v}`;
      const on = saved[idx] && saved[idx][d] == v ? " checked" : "";
      scale += `<label for="${id}"><input type="radio" id="${id}" name="q${idx}_${d}" value="${v}"${on}> ${v}</label>`;
    }
    rows += `<tr><td class="name">${d}</td><td class="anchor">1 = ${ANCHORS[d][0]}</td>`
          + `<td class="scale">${scale}</td><td class="anchor">5 = ${ANCHORS[d][1]}</td></tr>`;
  });
  div.innerHTML = `<div class="pos">Item ${idx + 1} of ${ROWS.length}</div>`
    + `<div class="profile"><strong>Customer:</strong> ${r.customer_profile}</div>`
    + `<div class="msg">${r.message_text}</div>`
    + `<table class="dims">${rows}</table>`;
  wrap.appendChild(div);
});

function refresh() {
  let done = 0;
  ROWS.forEach((r, idx) => {
    const vals = {};
    let full = true;
    DIMS.forEach(d => {
      const hit = document.querySelector(`input[name="q${idx}_${d}"]:checked`);
      if (hit) vals[d] = hit.value; else full = false;
    });
    if (Object.keys(vals).length) saved[idx] = vals;
    document.getElementById("item" + idx).classList.toggle("done", full);
    if (full) done++;
  });
  localStorage.setItem(KEY, JSON.stringify(saved));
  document.getElementById("status").textContent = done + " of " + ROWS.length + " scored";
  document.getElementById("dl").disabled = done !== ROWS.length;
}
document.addEventListener("change", e => { if (e.target.type === "radio") refresh(); });
refresh();

document.getElementById("dl").addEventListener("click", () => {
  const head = ["sheet_position","profile_id","message_label"].concat(DIMS);
  const lines = [head.join(",")];
  ROWS.forEach((r, idx) => {
    const v = saved[idx] || {};
    lines.push([r.sheet_position, r.profile_id, r.message_label]
               .concat(DIMS.map(d => v[d] || "")).join(","));
  });
  const blob = new Blob([lines.join("\\n")], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "rater___K__.csv";
  a.click();
});
</script></body></html>
"""


def main():
    outdir = config.PROCESSED_DIR / "rating_sheets"
    sheets = sorted(outdir.glob("rater_*.csv"))
    if not sheets:
        raise SystemExit("No rating sheets found. Run scripts/make_rating_sheets.py first.")

    for path in sheets:
        k = path.stem.split("_")[1]
        df = pd.read_csv(path)
        rows = [
            {
                "sheet_position": int(r.sheet_position),
                "profile_id": str(r.profile_id),
                "message_label": str(r.message_label),
                "customer_profile": html.escape(str(r.customer_profile)),
                "message_text": html.escape(str(r.message_text)),
            }
            for r in df.itertuples()
        ]
        page = (TEMPLATE
                .replace("__ROWS__", json.dumps(rows))
                .replace("__DIMS__", json.dumps(DIMENSIONS))
                .replace("__ANCHORS__", json.dumps(ANCHORS))
                .replace("__N__", str(len(rows)))
                .replace("__K__", k))
        out = outdir / f"rater_{k}.html"
        out.write_text(page, encoding="utf-8")
        print(f"  wrote {out.name}  ({len(rows)} items)")

    print(f"\nSend the .html files to your raters, one each. They open in any "
          f"browser,\nno install needed, and produce a rater_<k>.csv to send back. "
          f"Put the\nreturned CSVs in {outdir} (overwriting the blank templates), "
          f"then run\nscripts/score_retention_eval.py")


if __name__ == "__main__":
    main()

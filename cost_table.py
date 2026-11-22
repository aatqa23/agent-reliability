"""Cost per run: tokens, API calls and wall-clock time per episode, from the phase-3 log fields (no API calls).

  python cost_table.py                         # every results/**/*.jsonl that has token fields
  python cost_table.py --logs results/phase3/e4_pe_len4_v2.jsonl ...
  python cost_table.py --price openai/gpt-oss-120b=0.15,0.60      # optional: USD per 1M input,output tokens

Writes analysis/cost_per_run.csv and prints a Markdown table. Rows written before phase 3 have no token fields and
are skipped (they are counted in the "skipped" line). Infrastructure errors are dropped; the newest row per episode
wins, as in metrics.load_logs. results/smoke/ is ignored (test runs, partly with an older planner).

Columns: n = episodes; tokens = mean total tokens per episode (prompt + completion; cached replies count with the
usage they had when they ran live); in/out = mean prompt/completion tokens; calls = mean LLM calls; wall_s = mean
wall-clock seconds of LIVE episodes only (cached replays are not timed meaningfully); success = success rate;
tok_per_success = mean tokens x n / successful episodes (expected cost of one success); per_200k = episodes that fit in one
free-tier day (200,000 tokens per model). zero_tok = episodes logged with 0 tokens: every call was a rejected
generation (Groq 400 tool_use_failed), whose usage the API does not report; they are left out of the token means.
"""
import argparse
import glob
import json
import os

import pandas as pd

from metrics import EPISODE_KEY, is_infra

GROUP = ["agent", "model", "condition", "chain_len"]
FREE_TIER_DAILY = 200_000


def default_logs() -> list:
    files = glob.glob(os.path.join("results", "**", "*.jsonl"), recursive=True)
    return sorted(f for f in files if os.sep + "smoke" + os.sep not in f)


def load(paths: list) -> tuple[pd.DataFrame, int]:
    latest, skipped = {}, 0
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if is_infra(r.get("error", "")):
                    continue
                if r.get("total_tokens") is None:
                    skipped += 1
                    continue
                latest[tuple(r.get(k) for k in EPISODE_KEY)] = r
    return pd.DataFrame(list(latest.values())), skipped


def table(df: pd.DataFrame, prices: dict) -> pd.DataFrame:
    df = df[df["untracked_calls"].fillna(0) == 0]          # tokens unknown for old cache entries
    rows = []
    for g, s in df.groupby(GROUP, dropna=False):
        live = s[s["api_calls"] > 0]
        succ = int(s["success"].sum())
        c = s[s["total_tokens"] > 0]                          # token means leave out under-counted 0-token episodes
        tok = c["total_tokens"].mean() if len(c) else 0
        row = {**dict(zip(GROUP, g)), "n": len(s),
               "tokens": round(tok), "in": round(c["prompt_tokens"].mean()), "out": round(c["completion_tokens"].mean()),
               "calls": round((s["api_calls"] + s["cache_hits"]).mean(), 1),
               "wall_s": round(live["wall_clock_s"].mean(), 1) if len(live) else None,
               "success": round(s["success"].mean(), 3),
               "tok_per_success": round(tok * len(s) / succ) if succ else None,
               "per_200k": int(FREE_TIER_DAILY // tok) if tok else None,
               "zero_tok": int((s["total_tokens"] == 0).sum())}
        if g[1] in prices:
            pin, pout = prices[g[1]]
            row["usd_per_ep"] = round((c["prompt_tokens"].mean() * pin + c["completion_tokens"].mean() * pout) / 1e6, 5)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["chain_len", "agent", "model"]).reset_index(drop=True)


def to_markdown(t: pd.DataFrame) -> str:
    cols = list(t.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", default=None)
    ap.add_argument("--price", action="append", default=[], help="MODEL=IN,OUT  (USD per 1M tokens)")
    ap.add_argument("--out", default=os.path.join("analysis", "cost_per_run.csv"))
    a = ap.parse_args(argv)
    prices = {}
    for p in a.price:
        m, v = p.split("=")
        pin, pout = (float(x) for x in v.split(","))
        prices[m] = (pin, pout)
    paths = a.logs or default_logs()
    df, skipped = load(paths)
    if df.empty:
        raise SystemExit("no rows with token fields found")
    t = table(df, prices)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t.to_csv(a.out, index=False)
    print(to_markdown(t))
    print(f"\n{len(df)} episodes with token data; {skipped} older rows without token fields skipped -> {a.out}")


if __name__ == "__main__":
    main()

"""Replay one failed episode from the saved LLM cache (NO API calls) and show what the agent did.

  python inspect_episode.py results/e4_len4_120b.jsonl            # first failed episode in that file
  python inspect_episode.py results/e4_len4_120b.jsonl --nth 3    # the 3rd failed episode
  python inspect_episode.py results/e4_len4_120b.jsonl --ok       # look at a successful episode instead

Works because every reply was cached (key = messages + model + seed). If a reply is missing from the cache the script
stops instead of calling the API.
"""
import argparse
import copy
import json
import sys

import chains as chains_mod
import llm
from agents import AGENTS
from contract import get_path, load_base_db
from env.tools import TOOLS
from run_e1 import dedupe, read_logs
from taskset import load_all_tasks


def no_live_calls(*a, **k):
    raise RuntimeError("cache miss: this reply is not in results/cache (is this the folder where the run happened?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--nth", type=int, default=1)
    ap.add_argument("--ok", action="store_true", help="pick a successful episode instead of a failed one")
    a = ap.parse_args()

    rows = [r for r in dedupe(read_logs(a.file)) if bool(r["success"]) == a.ok and not r["error"].startswith(("api_error", "crash"))]
    if len(rows) < a.nth:
        sys.exit(f"only {len(rows)} matching episodes in {a.file}")
    r = rows[a.nth - 1]
    tasks = {t["id"]: t for t in load_all_tasks()}
    if "+" in r["task_id"]:
        tasks.update({c["id"]: c for c in chains_mod.load_chains()})
    task = tasks[r["task_id"]]
    instruction = task["instruction"]
    if r["condition"] == "paraphrase":
        instruction = task["paraphrases"][r["paraphrase_idx"]]

    llm._chat_live = no_live_calls
    db = copy.deepcopy(load_base_db())
    res = AGENTS[r["agent"]](instruction, db, TOOLS, r["model"], r["seed"], fault_hook=None,
                             max_steps=15 * max(1, r["chain_len"]))
    print(f"episode: {r['task_id']}  model={r['model']}  seed={r['seed']}  logged: success={r['success']} steps={r['steps']}")
    print(f"replay : steps={res['steps']} tool_calls={res['tool_calls']} error={res['error'] or '-'}\n")
    for m in res["trace"][1:]:
        text = m["content"].replace("\n", " ")
        print(f"[{m['role']:9}] {text[:400]}{'...' if len(text) > 400 else ''}")
    print("\nASSERTIONS (expected -> actual):")
    for asr in task["assertions"]:
        actual = get_path(db, asr["path"])
        print(f"  {'OK  ' if actual == asr['equals'] else 'FAIL'} {asr['path']}: expected {asr['equals']!r}, got {actual!r}")


if __name__ == "__main__":
    main()

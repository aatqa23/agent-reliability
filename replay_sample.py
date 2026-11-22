"""Small cached sample so episodes can be re-run OFFLINE (no API key, no network).

The full reply cache (results/cache/) is too large to ship. This script keeps a small one instead:
  results/sample/sample.jsonl   the logged episodes (normal EpisodeLog rows, incl. tokens and wall-clock time)
  results/sample/cache/         every model reply those episodes needed (same format as results/cache/)

  python replay_sample.py --check     # offline: re-runs every sample episode from the cache and checks that
                                      # success, steps, tool_calls, error, tokens and plan_ok match the log
  python replay_sample.py --record    # live (needs GROQ_API_KEY): (re)builds the sample; existing rows are kept

The same mechanism works for any run: point LLM_CACHE_DIR at a cache and set LLM_OFFLINE=1, e.g.
  LLM_CACHE_DIR=results/sample/cache LLM_OFFLINE=1 python run_e1.py ...
"""
import argparse
import json
import os
import sys

import llm
from contract import load_tasks
from run_e1 import key_of, read_logs
from run_episode import run_episode

SAMPLE_DIR = os.path.join("results", "sample")
SAMPLE_LOG = os.path.join(SAMPLE_DIR, "sample.jsonl")
SAMPLE_CACHE = os.path.join(SAMPLE_DIR, "cache")
TASK_IDS = ["A01", "A04", "C03"]                    # one simple act, a two-request task, a refusal (policy 4)
AGENTS = ["react", "plan_execute"]
MODELS = [llm.MODELS["large"], llm.MODELS["small"], llm.MODELS["qwen"]]
SEED = 0
COMPARE = ("success", "steps", "tool_calls", "error", "prompt_tokens", "completion_tokens", "total_tokens",
           "plan_ok")


def sample_tasks() -> list:
    tasks = {}
    for f in ("A", "B", "C"):
        for t in load_tasks(os.path.join("tasks", f"{f}.json")):
            tasks[t["id"]] = t
    return [tasks[i] for i in TASK_IDS]


def record() -> None:
    llm.CACHE_DIR = SAMPLE_CACHE
    done = {key_of(r) for r in read_logs(SAMPLE_LOG) if r.get("error") != "api_error"}
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    for task in sample_tasks():
        for agent in AGENTS:
            for model in MODELS:
                k = (task["id"], agent, model, "clean", None, 0.0, 1, SEED)
                if k in done:
                    continue
                d = run_episode(task, agent, model, None, SEED)
                with open(SAMPLE_LOG, "a") as f:
                    f.write(json.dumps(d) + "\n")
                print(f"{task['id']} {agent:<13}{model:<22} success={d['success']} steps={d['steps']} "
                      f"tokens={d['total_tokens']} wall={d['wall_clock_s']}s")


def check() -> int:
    os.environ["LLM_OFFLINE"] = "1"
    llm.CACHE_DIR = SAMPLE_CACHE
    rows = read_logs(SAMPLE_LOG)
    if not rows:
        print(f"FAIL: {SAMPLE_LOG} is missing or empty")
        return 1
    tasks = {t["id"]: t for t in sample_tasks()}
    bad = 0
    for r in rows:
        d = run_episode(tasks[r["task_id"]], r["agent"], r["model"], None, r["seed"])
        diff = {k: (r.get(k), d.get(k)) for k in COMPARE if r.get(k) != d.get(k)}
        if diff or d["api_calls"] != 0:
            bad += 1
            print(f"MISMATCH {r['task_id']} {r['agent']} {r['model']}: {diff} api_calls={d['api_calls']}")
    if bad:
        print(f"FAIL: {bad} of {len(rows)} sample episodes did not replay identically")
        return 1
    print(f"OK: {len(rows)} sample episodes replayed offline from the cache, all identical (0 API calls)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--record", action="store_true")
    a = ap.parse_args()
    if a.record:
        record()
    else:
        sys.exit(check())

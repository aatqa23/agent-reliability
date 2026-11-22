"""E1: clean-condition runs. Usage examples:

  python run_e1.py --limit 3 --n 2                  # smoke test: first 3 tasks, 2 trials each
  python run_e1.py                                  # full run: tasks/A.json, both models, n=5
  python run_e1.py --workers 1 --delay 1.0          # gentler on rate limits
  python run_e1.py --summary-only                   # just print the summary table

Safe to stop (Ctrl-C) and re-run: finished episodes are skipped. Episodes that ended in an
infrastructure error (api_error / crash) are retried on the next run; the newest line wins.
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from contract import EpisodeLog, append_log, load_tasks
from run_episode import run_episode

DEFAULT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
KEY_FIELDS = ("task_id", "agent", "model", "condition", "paraphrase_idx", "fault_rate", "chain_len", "seed")
CLEAN = {"name": "clean", "paraphrase_idx": None, "fault_rate": 0.0, "chain_len": 1}


def is_infra(err: str) -> bool:
    return err == "api_error" or err.startswith("crash")


def key_of(row: dict) -> tuple:
    return tuple(row.get(k) for k in KEY_FIELDS)


def read_logs(path: str) -> list:
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass          # ignore a half-written last line
    return rows


def dedupe(rows: list) -> list:
    """One row per episode key; the newest line wins."""
    latest = {}
    for r in rows:
        latest[key_of(r)] = r
    return list(latest.values())


def done_keys(path: str) -> set:
    return {key_of(r) for r in dedupe(read_logs(path)) if not is_infra(r.get("error", ""))}


def build_jobs(tasks, agents, models, n, done):
    jobs = []
    for seed in range(n):                      # seed outermost: partial runs stay balanced
        for task in tasks:
            for agent in agents:
                for model in models:
                    k = (task["id"], agent, model, "clean", None, 0.0, 1, seed)
                    if k not in done:
                        jobs.append((task, agent, model, seed))
    return jobs


def summarize(path: str) -> list:
    rows = dedupe(read_logs(path))
    groups = defaultdict(lambda: defaultdict(list))
    infra = defaultdict(int)
    for r in rows:
        g = (r["agent"], r["model"], r["condition"], r["fault_rate"], r["chain_len"])
        groups[g][r["task_id"]].append(bool(r["success"]))
        if is_infra(r.get("error", "")):
            infra[g] += 1
    out = []
    for g, by_task in sorted(groups.items()):
        trials = [len(v) for v in by_task.values()]
        k = min(trials)
        out.append({
            "agent": g[0], "model": g[1], "condition": g[2], "fault_rate": g[3], "chain_len": g[4],
            "tasks": len(by_task), "trials/task": k,
            "pass@1": sum(sum(v) / len(v) for v in by_task.values()) / len(by_task),
            "pass@n": sum(any(v) for v in by_task.values()) / len(by_task),
            "pass^n": sum(all(v) for v in by_task.values()) / len(by_task),
            "infra_errors": infra[g],
        })
    return out


def print_summary(path: str) -> None:
    rows = summarize(path)
    if not rows:
        print("(no results yet)")
        return
    print(f"\n{'agent':<13}{'model':<22}{'cond':<11}{'tasks':>6}{'n':>3}{'pass@1':>9}{'pass@n':>9}{'pass^n':>9}{'infra_err':>10}")
    for r in rows:
        print(f"{r['agent']:<13}{r['model']:<22}{r['condition']:<11}{r['tasks']:>6}{r['trials/task']:>3}"
              f"{r['pass@1']:>9.3f}{r['pass@n']:>9.3f}{r['pass^n']:>9.3f}{r['infra_errors']:>10}")
    if any(r["infra_errors"] for r in rows):
        print("\n[warn] some episodes ended in api_error/crash. Re-run the same command to retry them.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join("tasks", "A.json"))
    ap.add_argument("--agents", nargs="+", default=["react"])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--n", type=int, default=5, help="trials per task (seeds 0..n-1)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks (smoke test)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait before each episode")
    ap.add_argument("--out", default=os.path.join("results", "e1.jsonl"))
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)

    if a.summary_only:
        print_summary(a.out)
        return

    tasks = load_tasks(a.tasks)
    if a.limit:
        tasks = tasks[: a.limit]
    jobs = build_jobs(tasks, a.agents, a.models, a.n, done_keys(a.out))
    total = len(tasks) * len(a.agents) * len(a.models) * a.n
    print(f"{len(tasks)} tasks x {len(a.agents)} agents x {len(a.models)} models x {a.n} trials = {total} episodes; "
          f"{total - len(jobs)} already done, {len(jobs)} to run (workers={a.workers})")

    lock = threading.Lock()

    def work(job):
        task, agent, model, seed = job
        if a.delay:
            time.sleep(a.delay)
        try:
            d = run_episode(task, agent, model, CLEAN, seed)
            log = EpisodeLog(**d)
        except Exception as e:
            log = EpisodeLog(task_id=task["id"], agent=agent, model=model, condition="clean",
                             paraphrase_idx=None, fault_rate=0.0, chain_len=1, seed=seed, success=False,
                             steps=0, tool_calls=0, error=f"crash: {type(e).__name__}: {e}"[:200], latency_s=0.0)
        with lock:
            append_log(a.out, log)
        return log

    try:
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futures = [ex.submit(work, j) for j in jobs]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="episodes"):
                pass
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.")
        sys.exit(1)

    print_summary(a.out)


if __name__ == "__main__":
    main()

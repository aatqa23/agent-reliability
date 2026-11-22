"""B's experiment runner: E2 (paraphrases), E3 (faults), E4 (chains), plus the clean baseline for B's tasks.
Same log format, resume logic and parallelism as run_e1.py (it reuses its helpers), every episode goes through
run_episode(). Safe to stop (Ctrl-C) and re-run: finished episodes are skipped, api_error/crash rows are retried.

  python run_experiments.py clean --tasks tasks/B.json   --out results/e1_b.jsonl    # baseline for B01-B10
  python run_experiments.py e2    --workers 1 --delay 1      # 3 paraphrases x tasks x 2 models x 5 seeds -> results/e2.jsonl
  python run_experiments.py e3    --workers 1 --delay 1      # fault rates 0.1 and 0.2                    -> results/e3.jsonl
  python run_experiments.py e4    --workers 1 --delay 1      # chains of length 2,3,4 (data/chains.json)  -> results/e4.jsonl
  python run_experiments.py e2 --dry-run                     # count episodes and estimate time, run nothing
  python run_experiments.py e2 --limit 2 --n 1 --models openai/gpt-oss-20b     # smoke test
  python run_experiments.py summary results/e2.jsonl results/e3.jsonl results/e4.jsonl

By default E2/E3 use every task file that exists (tasks/A.json, B.json, C.json). When C.json arrives, just
re-run the same command: the episodes already done are skipped and only C's tasks are added.
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

import chains as chains_mod
import llm
from contract import EpisodeLog, append_log, load_base_db
from run_e1 import DEFAULT_MODELS, dedupe, is_infra, key_of, read_logs
from run_episode import run_episode
from taskset import TASK_FILES, load_all_tasks, inject_paraphrases

SECONDS_PER_STEP_UNIT = 12.0     # rough: A's E1 episodes took ~11.5 s on average (one single-task episode)
DEFAULTS = {
    "clean": os.path.join("results", "e1_b.jsonl"),
    "e2": os.path.join("results", "e2.jsonl"),
    "e3": os.path.join("results", "e3.jsonl"),
    "e4": os.path.join("results", "e4.jsonl"),
}


def cond(name, paraphrase_idx=None, fault_rate=0.0, chain_len=1) -> dict:
    return {"name": name, "paraphrase_idx": paraphrase_idx, "fault_rate": float(fault_rate), "chain_len": chain_len}


def units_for(exp: str, tasks: list, args) -> list:
    """The (task, condition) pairs an experiment needs; each gets one episode per agent x model x seed."""
    if exp == "clean":
        return [(t, cond("clean")) for t in tasks]
    if exp == "e2":
        units = []
        for t in tasks:
            if len(t.get("paraphrases", [])) < args.paraphrases:
                raise SystemExit(f"{t['id']} has no paraphrases: run `python paraphraser.py` first "
                                 f"(then `python paraphraser.py --check`).")
            units += [(t, cond("paraphrase", paraphrase_idx=i)) for i in range(args.paraphrases)]
        return units
    if exp == "e3":
        return [(t, cond("fault", fault_rate=r)) for t in tasks for r in args.rates]
    if exp == "e4":
        stale = chains_mod.stale_reason(args.chains)
        if stale:
            print(f"[warn] {stale}", file=sys.stderr)
        chs = chains_mod.load_chains(args.chains, args.lens)
        if args.per_len:                       # first K chains of each length (file order is frozen, so everyone gets the same ones)
            kept, count = [], defaultdict(int)
            for c in chs:
                L = len(c["chain"])
                if count[L] < args.per_len:
                    count[L] += 1
                    kept.append(c)
            chs = kept
        return [(c, cond("chain", chain_len=len(c["chain"]))) for c in chs]
    raise ValueError(exp)


def job_key(task, agent, model, c, seed) -> tuple:
    return (task["id"], agent, model, c["name"], c["paraphrase_idx"], c["fault_rate"], c["chain_len"], seed)


def build_jobs(units, agents, models, n, done):
    jobs = []
    for seed in range(n):                      # seed outermost: a partial run stays balanced across tasks
        for task, c in units:
            for agent in agents:
                for model in models:
                    if job_key(task, agent, model, c, seed) not in done:
                        jobs.append((task, c, agent, model, seed))
    return jobs


def done_keys(path):
    return {key_of(r) for r in dedupe(read_logs(path)) if not is_infra(r.get("error", ""))}


def summarize(path: str) -> list:
    """Per (agent, model, condition, fault_rate, chain_len): the unit of analysis is (task_id, paraphrase_idx), so
    paraphrase variants are never pooled into one task. pass@1 = mean success; pass@n / pass^n over the n trials."""
    rows = dedupe(read_logs(path))
    groups = defaultdict(lambda: defaultdict(list))
    extra = defaultdict(lambda: {"steps": 0, "tools": 0, "n": 0, "infra": 0, "errors": defaultdict(int)})
    for r in rows:
        g = (r["agent"], r["model"], r["condition"], r["fault_rate"], r["chain_len"])
        groups[g][(r["task_id"], r.get("paraphrase_idx"))].append(bool(r["success"]))
        e = extra[g]
        e["steps"] += r["steps"]; e["tools"] += r["tool_calls"]; e["n"] += 1
        e["infra"] += is_infra(r.get("error", ""))
        if r.get("error"):
            e["errors"][r["error"].split(":")[0]] += 1
    out = []
    for g, by_unit in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][3], kv[0][4])):
        e = extra[g]
        out.append({"agent": g[0], "model": g[1].split("/")[-1], "condition": g[2], "fault_rate": g[3], "chain_len": g[4],
                    "units": len(by_unit), "trials": min(len(v) for v in by_unit.values()),
                    "pass@1": sum(sum(v) / len(v) for v in by_unit.values()) / len(by_unit),
                    "pass@n": sum(any(v) for v in by_unit.values()) / len(by_unit),
                    "pass^n": sum(all(v) for v in by_unit.values()) / len(by_unit),
                    "steps": e["steps"] / e["n"], "tool_calls": e["tools"] / e["n"],
                    "infra": e["infra"], "errors": dict(e["errors"])})
    return out


def print_summary(path: str) -> None:
    rows = summarize(path)
    print(f"\n== {path}")
    if not rows:
        print("(no results yet)")
        return
    print(f"{'model':<14}{'cond':<11}{'fault':>6}{'chain':>6}{'units':>6}{'n':>3}{'pass@1':>8}{'pass@n':>8}{'pass^n':>8}{'steps':>7}{'infra':>6}  other errors")
    for r in rows:
        other = {k: v for k, v in r["errors"].items() if k not in ("api_error", "crash")}
        print(f"{r['model']:<14}{r['condition']:<11}{r['fault_rate']:>6.2f}{r['chain_len']:>6}{r['units']:>6}{r['trials']:>3}"
              f"{r['pass@1']:>8.3f}{r['pass@n']:>8.3f}{r['pass^n']:>8.3f}{r['steps']:>7.1f}{r['infra']:>6}  {other or ''}")
    if any(r["infra"] for r in rows):
        print("[warn] some episodes ended in api_error/crash. Re-run the same command to retry them.")


def merge_files(files: list, out: str, only_complete: int | None) -> None:
    """Combine result files into one: infra-error rows dropped, newest row per episode wins, and (optionally) only
    units (task, agent, model, condition, paraphrase, fault rate, chain length) that have ALL seeds 0..N-1 are kept,
    so every unit in the merged file has the same number of trials."""
    rows = []
    for f in files:
        rows += read_logs(f)
    rows = [r for r in dedupe(rows) if not is_infra(r.get("error", ""))]
    units = defaultdict(set)
    for r in rows:
        units[tuple(r.get(k) for k in rx_UNIT_FIELDS)].add(r["seed"])
    if only_complete:
        full = {u for u, seeds in units.items() if set(range(only_complete)) <= seeds}
        dropped = len(units) - len(full)
        rows = [r for r in rows if tuple(r.get(k) for k in rx_UNIT_FIELDS) in full]
        rows = [r for r in rows if r["seed"] < only_complete]
        print(f"kept {len(full)} complete units, dropped {dropped} incomplete units")
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; move or delete it first (merge never overwrites)")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} episodes to {out}")


rx_UNIT_FIELDS = ("task_id", "agent", "model", "condition", "paraphrase_idx", "fault_rate", "chain_len")


def estimate_seconds(jobs, workers, delay) -> float:
    return sum(SECONDS_PER_STEP_UNIT * max(1, c["chain_len"]) + delay for _, c, _, _, _ in jobs) / max(1, workers)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", choices=["clean", "e2", "e3", "e4", "summary", "merge"])
    ap.add_argument("files", nargs="*", help="(summary / merge) result files")
    ap.add_argument("--tasks", nargs="+", default=None, help="task files (default: tasks/A.json B.json C.json that exist)")
    ap.add_argument("--agents", nargs="+", default=["react"])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--n", type=int, default=5, help="trials per unit (seeds 0..n-1)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N units (smoke test)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait before each episode")
    ap.add_argument("--out", default=None)
    ap.add_argument("--paraphrases", type=int, default=3, help="(e2) paraphrases per task")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2], help="(e3) fault rates")
    ap.add_argument("--chains", default=chains_mod.CHAINS_PATH, help="(e4) frozen chains file")
    ap.add_argument("--lens", type=int, nargs="+", default=None, help="(e4) only these chain lengths")
    ap.add_argument("--per-len", type=int, default=None, help="(e4) only the first K chains of each length")
    ap.add_argument("--only-complete", type=int, default=None, help="(merge) keep only units that have all seeds 0..N-1")
    ap.add_argument("--dry-run", action="store_true", help="count episodes and estimate time, run nothing")
    a = ap.parse_args(argv)

    if a.exp == "summary":
        for f in a.files or [DEFAULTS[k] for k in ("clean", "e2", "e3", "e4")]:
            print_summary(f)
        return

    if a.exp == "merge":
        if not a.out or not a.files:
            raise SystemExit("usage: run_experiments.py merge FILE [FILE ...] --out results/e3.jsonl [--only-complete 5]")
        merge_files(a.files, a.out, a.only_complete)
        return

    out = a.out or DEFAULTS[a.exp]
    tasks = load_all_tasks([*a.tasks] if a.tasks else TASK_FILES)
    if not tasks and a.exp != "e4":
        raise SystemExit("no tasks found (run from the project root; tasks/*.json)")
    units = units_for(a.exp, tasks, a)
    if a.limit:
        units = units[: a.limit]
    done = done_keys(out)
    jobs = build_jobs(units, a.agents, a.models, a.n, done)
    total = len(units) * len(a.agents) * len(a.models) * a.n
    est = estimate_seconds(jobs, a.workers, a.delay)
    print(f"{a.exp}: {len(units)} units x {len(a.agents)} agents x {len(a.models)} models x {a.n} trials = {total} episodes; "
          f"{total - len(jobs)} already done, {len(jobs)} to run (workers={a.workers}) -> {out}")
    print(f"rough time for the remaining episodes: {est / 3600:.1f} h (assumes ~{SECONDS_PER_STEP_UNIT:.0f} s per single-task episode; "
          f"real time depends on your rate limit)")
    if a.dry_run or not jobs:
        return

    lock = threading.Lock()

    def work(job):
        task, c, agent, model, seed = job
        if a.delay:
            time.sleep(a.delay)
        try:
            log = EpisodeLog(**run_episode(task, agent, model, c, seed))
        except Exception as e:
            log = EpisodeLog(task_id=task["id"], agent=agent, model=model, condition=c["name"],
                             paraphrase_idx=c["paraphrase_idx"], fault_rate=c["fault_rate"], chain_len=c["chain_len"],
                             seed=seed, success=False, steps=0, tool_calls=0,
                             error=f"crash: {type(e).__name__}: {e}"[:200], latency_s=0.0)
        with lock:
            append_log(out, log)
        return log

    try:
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futures = [ex.submit(work, j) for j in jobs]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=a.exp):
                pass
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.")
        sys.exit(1)

    s = llm.cache_stats()
    print(f"LLM cache: {s['hits']} hits, {s['misses']} live calls this run")
    print_summary(out)


if __name__ == "__main__":
    main()

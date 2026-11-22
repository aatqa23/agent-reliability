"""Offline tests for B's components (no API calls): task loading, paraphraser, chains, fault wiring, E2-E4 runner.
Run: python test_b.py
Uses an 'oracle' agent that solves any task straight from its assertions (retrying through injected faults),
so every failure here is a bug in the pipeline, not model noise."""
import copy
import json
import os
import random
import re
import tempfile

import chains as chains_mod
import paraphraser as pz
import run_experiments as rx
from agents import AGENTS
from contract import call_tool, check_assertions, load_base_db, load_tasks, validate_task, get_path
from env.tools import TOOLS
from run_e1 import read_logs
from run_episode import run_episode
from taskset import load_all_tasks, load_paraphrases


def expect(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    assert cond, label


base = load_base_db()
tasks = load_all_tasks()
by_id = {t["id"]: t for t in tasks}

# ------------------------------------------------------------------ task loading / B.json
expect("A+B tasks load (20)", len(tasks) == 20 and [t["id"] for t in tasks][:1] == ["A01"])
expect("paraphrases injected into every task (3 each)", all(len(t["paraphrases"]) == 3 for t in tasks))
expect("all B tasks valid", not any(validate_task(t, base) for t in load_tasks("tasks/B.json")))
expect("paraphrases.json passes the validator", pz.check_all(load_all_tasks(with_paraphrases=False), load_paraphrases()) == [])
bt = load_tasks("tasks/B.json")
expect("B has >= 2 refusals and >= 2 multi-assertion tasks",
       sum(bool(t.get("refusal")) for t in bt) >= 2 and sum(len(t["assertions"]) >= 2 for t in bt) >= 2)

# ------------------------------------------------------------------ paraphrase validator
o = by_id["B07"]["instruction"]
good = by_id["B07"]["paraphrases"][0]
expect("validator accepts a good paraphrase", pz.validate_paraphrase(o, good) == [])
expect("rejects empty", pz.validate_paraphrase(o, "  ") == ["empty"])
expect("rejects identical", any("identical" in e for e in pz.validate_paraphrase(o, o.upper())))
expect("rejects missing order id", any("missing ids" in e for e in pz.validate_paraphrase(o, good.replace("O1008", "my other order"))))
expect("rejects new order id", any("new ids" in e for e in pz.validate_paraphrase(o, good + " Also O1004.")))
expect("rejects changed amount", any("numbers" in e for e in pz.validate_paraphrase(o, good.replace("50", "55"))))
expect("rejects changed address", any("addresses" in e for e in pz.validate_paraphrase(o, good.replace("Phase 4", "Phase 5"))))
expect("rejects dropped customer id", any("missing ids" in e for e in pz.validate_paraphrase(o, good.replace("C4", ""))))
expect("rejects list-number digits", any("new numbers" in e for e in pz.validate_paraphrase(o, "1) " + good)))
expect("validate_set rejects duplicates / wrong count",
       "duplicate paraphrases" in pz.validate_set(by_id["B07"], [good, good, good])
       and any("expected 3" in e for e in pz.validate_set(by_id["B07"], [good])))

# generate() with a fake LLM: recovers from garbage / invalid answers, keeps valid ones across tries
t = by_id["B03"]
replies = iter(["sorry, I cannot", json.dumps(["Refund 99 please", t["paraphrases"][0]]),
                "```json\n" + json.dumps(t["paraphrases"][1:] + ["x"]) + "\n```"])
out = pz.generate(t, chat_fn=lambda *a, **k: next(replies))
expect("generate: collects 3 valid paraphrases over several tries", out == t["paraphrases"])
try:
    pz.generate(t, chat_fn=lambda *a, **k: "[]", tries=3)
    expect("generate: raises if it can never produce valid ones", False)
except RuntimeError:
    expect("generate: raises if it can never produce valid ones", True)
msgs = pz.build_prompt(t)
expect("prompt contains the instruction", t["instruction"] in msgs[1]["content"])

# ------------------------------------------------------------------ chains
all_chains = chains_mod.load_chains()
expect("data/chains.json: 10 chains each of length 2,3,4",
       [sum(len(c["chain"]) == L for c in all_chains) for L in (2, 3, 4)] == [10, 10, 10])
expect("frozen chains pass check_chains", chains_mod.check_chains(all_chains, tasks, base) == [])
expect("frozen chains are not stale", chains_mod.stale_reason() is None)
expect("every chain: members touch disjoint orders",
       all(len(set().union(*[chains_mod.orders_of(by_id[i]) for i in c["chain"]])) ==
           sum(len(chains_mod.orders_of(by_id[i])) for i in c["chain"]) for c in all_chains))
expect("chain ids join members with '+', owner = first member's owner",
       all(c["id"] == "+".join(c["chain"]) and c["owner"] == by_id[c["chain"][0]]["owner"] for c in all_chains))
a1, a5, a9 = by_id["A01"], by_id["A05"], by_id["A09"]
expect("tasks sharing an order are incompatible (A05 and A09 both use O1008)", not chains_mod.compatible(a5, a9, base))
expect("tasks on different orders are compatible", chains_mod.compatible(a1, by_id["A02"], base))
# identity clash: A05 = Bilal (C2) asks about Omar's (C4) order O1008, A07 is spoken by Omar on another order
expect("A05 (C2 asks about C4's order) is never chained with A07 (spoken by C4), although their orders differ",
       not (chains_mod.orders_of(by_id["A05"]) & chains_mod.orders_of(by_id["A07"])) and not chains_mod.compatible(by_id["A05"], by_id["A07"], base))
expect("A06 (C5 asks about C3's order) is never chained with A03 (spoken by C3)",
       not (chains_mod.orders_of(by_id["A06"]) & chains_mod.orders_of(by_id["A03"])) and not chains_mod.compatible(by_id["A06"], by_id["A03"], base))
expect("no frozen chain contains a clashing pair",
       not any(("A05" in c["chain"] and "A07" in c["chain"]) or ("A06" in c["chain"] and "A03" in c["chain"]) for c in all_chains))
c = chains_mod.compose([by_id["B04"], by_id["B05"]])
expect("all-refusal chain gets refusal=true; instruction keeps members verbatim",
       c.get("refusal") is True and by_id["B04"]["instruction"] in c["instruction"] and by_id["B05"]["instruction"] in c["instruction"])
expect("chain wrapper tells the agent not to finish after a refusal", "do not call finish until" in c["instruction"])
expect("mixed chain is not marked refusal", "refusal" not in chains_mod.compose([by_id["B01"], by_id["B04"]]))
expect("composed chain passes the contract validator", validate_task(chains_mod.compose([by_id["B01"], by_id["B04"], by_id["A03"]]), base) == [])
b1 = chains_mod.build_chains(tasks, base, per_len=4, seed=7)
b2 = chains_mod.build_chains(tasks, base, per_len=4, seed=7)
b3 = chains_mod.build_chains(tasks, base, per_len=4, seed=8)
expect("build is deterministic for a fixed seed, differs for another", b1 == b2 and b1 != b3)
expect("stale_reason detects a changed task set", chains_mod.stale_reason(tasks=tasks[:-1]) is not None)
bad = copy.deepcopy(all_chains[0]); bad["assertions"] = bad["assertions"][:-1]
expect("check_chains catches a tampered chain", chains_mod.check_chains([bad], tasks, base) != [])

# ------------------------------------------------------------------ oracle agent (solves from assertions, retries through faults)
INSTR = {}
for t in tasks + all_chains:
    INSTR[t["instruction"]] = t
    for p in t.get("paraphrases", []):
        INSTR[p] = t
seen = []


def oracle(instruction, db, tools, model, seed, fault_hook=None, max_steps=15, chat_fn=None):
    seen.append({"instruction": instruction, "max_steps": max_steps})
    task = INSTR[instruction]
    n = 0
    for a in task["assertions"]:
        _, oid, field = a["path"].split(".")
        cur = get_path(db, a["path"])
        if cur == a["equals"]:
            continue
        if field == "status":
            call = ("cancel_order", {"order_id": oid, "reason": "oracle"})
        elif field == "address":
            call = ("modify_order_address", {"order_id": oid, "address": a["equals"]})
        else:
            call = ("refund_order", {"order_id": oid, "amount": a["equals"] - cur})
        for _ in range(12):                      # retry through injected faults
            n += 1
            if call_tool(tools, call[0], db, call[1], fault_hook).startswith("OK"):
                break
    return {"steps": n + 1, "tool_calls": n, "error": "", "trace": []}


AGENTS["oracle"] = oracle
expect("oracle solves every single task and every frozen chain (assertions are consistent)",
       all(check_assertions((lambda d: (oracle(t["instruction"], d, TOOLS, "m", 0), d)[1])(copy.deepcopy(base)), t["assertions"])
           for t in tasks + all_chains))

# ------------------------------------------------------------------ run_episode: step budget grows with chain length
seen.clear()
run_episode(by_id["A01"], "oracle", "m", None, 0)
run_episode(all_chains[10], "oracle", "m", {"name": "chain", "chain_len": 3}, 0)
expect("max_steps = 15 for a single task, 45 for chain_len 3", [s["max_steps"] for s in seen] == [15, 45])

# ------------------------------------------------------------------ experiments end to end
tmp = tempfile.mkdtemp()


def run(exp, *extra, tag=""):
    out = os.path.join(tmp, f"{exp}{tag}.jsonl")
    rx.main([exp, "--agents", "oracle", "--models", "m1", "m2", "--workers", "3", "--out", out, *extra])
    return out, read_logs(out)


# E2
seen.clear()
out, rows = run("e2", "--n", "2")
expect("e2: 20 tasks x 3 paraphrases x 2 models x 2 seeds = 240 episodes", len(rows) == 240)
expect("e2: logged condition/paraphrase_idx", all(r["condition"] == "paraphrase" and r["paraphrase_idx"] in (0, 1, 2) for r in rows))
expect("e2: every episode used the matching paraphrase text",
       {s["instruction"] for s in seen} == {p for t in tasks for p in t["paraphrases"]})
expect("e2: all succeed with the oracle", all(r["success"] and r["error"] == "" for r in rows))
n0 = len(rows)
rx.main(["e2", "--agents", "oracle", "--models", "m1", "m2", "--workers", "3", "--out", out, "--n", "2"])
expect("e2: re-run resumes (nothing re-run)", len(read_logs(out)) == n0)
rx.main(["e2", "--agents", "oracle", "--models", "m1", "m2", "--workers", "3", "--out", out, "--n", "3"])
expect("e2: raising --n only adds the new seed", len(read_logs(out)) == n0 + 120)

# E3
out, rows = run("e3", "--n", "5")
expect("e3: 20 tasks x 2 rates x 2 models x 5 seeds = 400 episodes", len(rows) == 400)
expect("e3: fault_rate logged as 0.1 / 0.2", {r["fault_rate"] for r in rows} == {0.1, 0.2} and all(r["condition"] == "fault" for r in rows))
clean_calls = sum(r["tool_calls"] for r in rows if r["fault_rate"] == 0.1)
heavy_calls = sum(r["tool_calls"] for r in rows if r["fault_rate"] == 0.2)
expect("e3: more faults -> more tool calls (oracle retries)", heavy_calls > clean_calls > 0)
expect("e3: faults are reproducible (same episode, same tool_calls on a second pass)",
       run_episode(by_id["B08"], "oracle", "m1", rx.cond("fault", fault_rate=0.2), 3)["tool_calls"] ==
       run_episode(by_id["B08"], "oracle", "m1", rx.cond("fault", fault_rate=0.2), 3)["tool_calls"])
expect("e3: --rates overrides", len(run("e3", "--n", "1", "--rates", "0.5", "--limit", "3", tag="_r")[1]) == 3 * 2)

# E4
seen.clear()
out, rows = run("e4", "--n", "2")
expect("e4: 30 chains x 2 models x 2 seeds = 120 episodes", len(rows) == 120)
expect("e4: chain_len 2/3/4 logged, task ids are chain ids",
       {r["chain_len"] for r in rows} == {2, 3, 4} and all("+" in r["task_id"] and r["condition"] == "chain" for r in rows))
expect("e4: step budget scaled per chain length",
       all(s["max_steps"] == 15 * len(INSTR[s["instruction"]]["chain"]) for s in seen))
expect("e4: --lens 4 selects only length-4 chains", {r["chain_len"] for r in run("e4", "--n", "1", "--lens", "4", tag="_l4")[1]} == {4})

# E4 subset: first K chains of each length
out, rows = run("e4", "--n", "1", "--lens", "2", "4", "--per-len", "3", tag="_pl")
expect("e4 --lens 2 4 --per-len 3 -> 3 chains of length 2 and 3 of length 4 (x 2 models)",
       len(rows) == 12 and {r["chain_len"] for r in rows} == {2, 4}
       and len({r["task_id"] for r in rows if r["chain_len"] == 2}) == 3)
expect("--per-len picks the same chains for everyone (first K in the frozen file)",
       {r["task_id"] for r in rows if r["chain_len"] == 2} == {c["id"] for c in all_chains if len(c["chain"]) == 2}.__class__(
           [c["id"] for c in all_chains if len(c["chain"]) == 2][:3]))

# merge: drops infra rows, keeps only complete units, never overwrites
ma, mb, mo = (os.path.join(tmp, n) for n in ("ma.jsonl", "mb.jsonl", "merged.jsonl"))
def mrow(task, seed, model, err=""):
    return json.dumps({"task_id": task, "agent": "react", "model": model, "condition": "fault", "paraphrase_idx": None,
                       "fault_rate": 0.1, "chain_len": 1, "seed": seed, "success": True, "steps": 3, "tool_calls": 2,
                       "error": err, "latency_s": 1.0})
with open(ma, "w") as f:
    for s_ in range(5):
        f.write(mrow("B01", s_, "m1") + "\n")                 # complete unit
    for s_ in range(3):
        f.write(mrow("A01", s_, "m1") + "\n")                 # incomplete unit (3 of 5 seeds)
    f.write(mrow("B02", 0, "m1", "api_error") + "\n")         # infra error row
with open(mb, "w") as f:
    for s_ in range(5):
        f.write(mrow("B01", s_, "m2") + "\n")
    f.write(mrow("B02", 0, "m1") + "\n")                      # a later good row replaces the api_error
rx.main(["merge", ma, mb, "--out", mo, "--only-complete", "5"])
mrows = read_logs(mo)
expect("merge: complete units kept (B01 on m1 and m2), incomplete A01 and 1-seed B02 dropped", len(mrows) == 10 and {r["task_id"] for r in mrows} == {"B01"})
try:
    rx.main(["merge", ma, "--out", mo])
    expect("merge refuses to overwrite", False)
except SystemExit:
    expect("merge refuses to overwrite", True)

# clean baseline for B
out, rows = run("clean", "--tasks", "tasks/B.json", "--n", "5")
expect("clean: 10 B tasks x 2 models x 5 seeds = 100 episodes, condition clean",
       len(rows) == 100 and all(r["condition"] == "clean" and r["paraphrase_idx"] is None and r["fault_rate"] == 0.0 for r in rows))

# crash handling: unknown agent is logged as crash, run continues
out = os.path.join(tmp, "crash.jsonl")
rx.main(["e3", "--agents", "nonexistent", "--models", "m1", "--n", "1", "--limit", "2", "--out", out])
expect("crash rows logged and flagged for retry", len(read_logs(out)) == 2 and all(r["error"].startswith("crash") for r in read_logs(out)))

# a task without paraphrases stops E2 with a clear message
try:
    t0 = copy.deepcopy(tasks[0]); t0["paraphrases"] = []
    rx.units_for("e2", [t0], type("A", (), {"paraphrases": 3})())
    expect("e2 refuses tasks without paraphrases", False)
except SystemExit as e:
    expect("e2 refuses tasks without paraphrases", "paraphraser.py" in str(e))

# summary: unit = (task, paraphrase_idx); pass@1 / pass@n / pass^n
log = os.path.join(tmp, "sum.jsonl")
def row(task, idx, seed, ok, model="openai/gpt-oss-20b"):
    return json.dumps({"task_id": task, "agent": "react", "model": model, "condition": "paraphrase", "paraphrase_idx": idx,
                       "fault_rate": 0.0, "chain_len": 1, "seed": seed, "success": ok, "steps": 3, "tool_calls": 2,
                       "error": "", "latency_s": 1.0})
with open(log, "w") as f:
    for s in range(4):
        f.write(row("A01", 0, s, True) + "\n")                  # always right
        f.write(row("A01", 1, s, s % 2 == 0) + "\n")           # right half the time (same task, other paraphrase)
s = rx.summarize(log)
expect("summary: paraphrases are separate units (2 units, not 1 task with 8 trials)", len(s) == 1 and s[0]["units"] == 2 and s[0]["trials"] == 4)
expect("summary: pass@1 = 0.75, pass@n = 1.0, pass^n = 0.5", abs(s[0]["pass@1"] - 0.75) < 1e-9 and s[0]["pass@n"] == 1.0 and s[0]["pass^n"] == 0.5)

print("\nAll B tests passed.")

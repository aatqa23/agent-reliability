"""Tests for run_episode and the E1 runner using a stub agent (no API calls). Run: python test_run.py"""
import json
import os
import sys
import tempfile
import types

from agents import AGENTS
from env.tools import TOOLS
from contract import call_tool
import run_episode as re_mod
import run_e1


def expect(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    assert cond, label


seen = {}


def stub(instruction, db, tools, model, seed, fault_hook=None, max_steps=15, chat_fn=None):
    """Cancels O1001 on even seeds, does nothing on odd seeds."""
    seen["instruction"], seen["hook"] = instruction, fault_hook
    n = 0
    if seed % 2 == 0:
        call_tool(tools, "cancel_order", db, {"order_id": "O1001", "reason": "t"}, fault_hook)
        n = 1
    return {"steps": 1, "tool_calls": n, "error": "", "trace": []}


AGENTS["stub"] = stub

task = {"id": "A01", "owner": "A", "instruction": "cancel O1001",
        "assertions": [{"path": "orders.O1001.status", "equals": "cancelled"}],
        "paraphrases": ["please drop order O1001"]}

# --- run_episode
d = re_mod.run_episode(task, "stub", "m1", None, 0)
expect("success on even seed", d["success"] is True)
expect("log fields", d["task_id"] == "A01" and d["condition"] == "clean" and d["chain_len"] == 1 and d["error"] == "")
d = re_mod.run_episode(task, "stub", "m1", None, 1)
expect("failure on odd seed", d["success"] is False)

d = re_mod.run_episode(task, "stub", "m1", {"name": "paraphrase", "paraphrase_idx": 0}, 0)
expect("paraphrase instruction used", seen["instruction"] == "please drop order O1001" and d["condition"] == "paraphrase")
try:
    re_mod.run_episode(task, "stub", "m1", {"name": "paraphrase", "paraphrase_idx": 5}, 0)
    expect("missing paraphrase raises", False)
except ValueError:
    expect("missing paraphrase raises ValueError", True)

fake = types.ModuleType("faults")
fake.make_fault_hook = lambda rate, rng: (lambda name, args: "ERROR: injected")
sys.modules["faults"] = fake
d = re_mod.run_episode(task, "stub", "m1", {"name": "fault", "fault_rate": 0.2}, 0)
expect("fault hook passed to agent", seen["hook"] is not None and d["fault_rate"] == 0.2)
expect("injected fault blocks the tool", d["success"] is False)
del sys.modules["faults"]

# state must not leak between episodes
d1 = re_mod.run_episode(task, "stub", "m1", None, 0)
d2 = re_mod.run_episode(task, "stub", "m1", None, 2)
expect("db reset between episodes", d1["success"] and d2["success"])

# --- E1 runner
tmp = tempfile.mkdtemp()
tasks_path = os.path.join(tmp, "tasks.json")
t2 = dict(task, id="A02")
with open(tasks_path, "w") as f:
    json.dump([task, t2], f)
out = os.path.join(tmp, "e1.jsonl")
argv = ["--tasks", tasks_path, "--agents", "stub", "--models", "m1", "m2", "--n", "4", "--out", out, "--workers", "3"]

run_e1.main(argv)
rows = run_e1.read_logs(out)
expect("2 tasks x 2 models x 4 trials = 16 lines", len(rows) == 16)

run_e1.main(argv)
expect("resume skips finished episodes", len(run_e1.read_logs(out)) == 16)

s = run_e1.summarize(out)
expect("summary per model", len(s) == 2 and all(r["tasks"] == 2 and r["trials/task"] == 4 for r in s))
expect("pass@1 = 0.5, pass@n = 1.0, pass^n = 0.0",
       all(abs(r["pass@1"] - 0.5) < 1e-9 and r["pass@n"] == 1.0 and r["pass^n"] == 0.0 for r in s))

# infra errors are retried and the newest line wins
with open(out, "a") as f:
    bad = dict(rows[0], error="api_error", success=False)
    f.write(json.dumps(bad) + "\n")
expect("infra error not counted as done", run_e1.key_of(bad) not in run_e1.done_keys(out))
run_e1.main(argv)
latest = [r for r in run_e1.dedupe(run_e1.read_logs(out)) if run_e1.key_of(r) == run_e1.key_of(bad)][0]
expect("retried episode replaced the error", latest["error"] == "")

# a crash inside run_episode is logged, not fatal
out2 = os.path.join(tmp, "e2.jsonl")
run_e1.main(["--tasks", tasks_path, "--agents", "nonexistent", "--models", "m1", "--n", "1", "--out", out2])
crash = run_e1.read_logs(out2)
expect("crash logged and flagged", len(crash) == 2 and all(r["error"].startswith("crash") for r in crash))

# smoke-test limit
out3 = os.path.join(tmp, "e3.jsonl")
run_e1.main(["--tasks", tasks_path, "--agents", "stub", "--models", "m1", "--n", "2", "--limit", "1", "--out", out3])
expect("--limit 1 runs one task", len(run_e1.read_logs(out3)) == 2)

print("\nAll runner tests passed.")

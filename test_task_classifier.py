"""Offline tests for task_classifier.py. Run from the project root: python test_task_classifier.py"""
import json
import os
import tempfile

import numpy as np

import chains as chains_mod
import metrics as M
import task_classifier as T
from contract import load_base_db
from taskset import load_all_tasks

base = load_base_db()
tasks = load_all_tasks()
by_id = {t["id"]: t for t in tasks}


def expect(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    assert cond, label


# 1) descriptors come from the task alone and match what the tasks say
f = {k: T.task_features(by_id[k], base) for k in ("A05", "C02", "C03", "C04", "C05", "C06", "C10", "A01")}
expect("wrong-customer requests are flagged (A05, C02, C03) and own-order requests are not (A01, C05)",
       all(f[k]["wrong_customer"] for k in ("A05", "C02", "C03")) and not any(f[k]["wrong_customer"] for k in ("A01", "C05")))
expect("social-pressure wording is flagged for C02/C03 only", f["C02"]["social_pressure"] and f["C03"]["social_pressure"]
       and not f["C05"]["social_pressure"] and not f["A05"]["social_pressure"])
expect("amount near the policy-3 limit: C04 (101) and C05 (100), not A01", f["C04"]["near_limit"] and f["C05"]["near_limit"] and not f["A01"]["near_limit"])
expect("'half' needs a lookup (C06)", f["C06"]["needs_lookup"] == 1)
expect("refusal parts: pure refusals have 1 and no state change; C10 has 3 state changes",
       f["C04"]["refusal_parts"] == 1 and f["C04"]["state_changes"] == 0 and f["C10"]["state_changes"] == 3)

# 2) chain descriptors aggregate the members
ch = {"id": "C10+A05", "chain": ["C10", "A05"]}
cf = T.chain_features(ch, by_id, base)
expect("chain: counts add, flags take the maximum, length is recorded",
       cf["n_assertions"] == 4 and cf["state_changes"] == 3 and cf["wrong_customer"] == 1 and cf["chain_len"] == 2)

# 3) units and labels from synthetic logs
rows = []
for tid, outcomes in {"A01": [True] * 5, "A05": [True, True, True, True, False]}.items():
    for s, ok in enumerate(outcomes):
        rows.append({"task_id": tid, "agent": "react", "model": "openai/gpt-oss-20b", "condition": "clean",
                     "paraphrase_idx": None, "fault_rate": 0.0, "chain_len": 1, "seed": s, "success": ok, "steps": 3,
                     "tool_calls": 2, "error": "", "latency_s": 1.0})
fd, p = tempfile.mkstemp(suffix=".jsonl")
with os.fdopen(fd, "w") as fh:
    fh.write("\n".join(json.dumps(r) for r in rows))
u = T.build_units(M.load_logs([p]), tasks, chains_mod.load_chains(), base)
lab = u.set_index("task_id")["unreliable"].to_dict()
expect("label = at least one failed trial", lab == {"A01": 0, "A05": 1})

# 4) the split is strictly by task: a training unit never shares a task with a held-out unit
logs = [os.path.join("results", f) for f in sorted(os.listdir("results")) if f.endswith(".jsonl")]
units = T.build_units(M.load_logs(logs), tasks, chains_mod.load_chains(), base)
sets = T.member_sets(units)
seed, test_groups, tr, te = T.fixed_split(units)
held = set().union(*[s for s, t in zip(sets, te) if t])
expect("fixed split: train/test share no task (chains included)", not any(s & held for s, t in zip(sets, tr) if t))
expect("fixed split: test has both classes and chains", units.loc[te, "unreliable"].nunique() == 2 and units.loc[te, "condition"].eq("chain").any())
# leave-one-task-out leakage check on one held-out single task that is also inside chains
g = "A05"
te_g = (units["task_id"] == g).to_numpy()
tr_g = np.array([g not in s for s in sets])
expect("LOTO: holding out A05 also removes every chain that contains A05",
       not any("A05" in s for s, t in zip(sets, tr_g) if t) and (~tr_g & ~te_g).sum() > 0)
p1 = T.loto(units, T.FEATURE_SETS["context+task"], "logreg")
expect("LOTO returns one probability per unit in [0, 1]", len(p1) == len(units) and p1.min() >= 0 and p1.max() <= 1)
print("\nAll task-classifier tests passed.")

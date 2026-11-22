"""Numbers for the paper that the standard scripts do not print directly (no API calls).

  python paper_numbers.py

* Table I/II: valid episodes, mean steps and tool calls per agent, model and condition.
* ReAct vs plan-then-execute on the SAME units and the SAME seeds (0-4): ReAct has 10 seeds on E2-E4 now,
  plan-then-execute has 5, so ReAct is cut to seeds 0-4 before pairing.
* pass^10 for the ReAct cells that have 10 runs.
"""
import numpy as np
import pandas as pd

from metrics import default_logs, load_logs, pass_at_k, pass_hat_k, unit_table

df = load_logs(default_logs("."))
df["model"] = df["model"].str.replace("openai/", "", regex=False)
print(len(df), "valid episodes")


def cond(r):
    if r["condition"] == "chain":
        return f"chain L{r['chain_len']}"
    if r["condition"] == "fault":
        return f"fault {r['fault_rate']}"
    return r["condition"]


df["cond"] = df.apply(cond, axis=1)
print("\n== episodes / steps / tool calls")
print(df.groupby(["agent", "model", "cond"]).agg(episodes=("success", "size"), steps=("steps", "mean"),
                                                   tool_calls=("tool_calls", "mean"), success=("success", "mean"),
                                                   seeds=("seed", "nunique")).round(2).to_string())
if "plan_ok" in df:
    pe = df[df["agent"] == "plan_execute"]
    print("\nplan_ok:", pe["plan_ok"].value_counts(dropna=False).to_dict())

print("\n== pass^10 (ReAct cells with 10 runs)")
u = unit_table(df[df["agent"] == "react"])
for (m, c, fr, L), s in u.groupby(["model", "condition", "fault_rate", "chain_len"]):
    if s["n"].min() >= 10:
        print(m, c, fr, L, "units", len(s), "pass@1 %.3f pass^5 %.3f pass^10 %.3f" % (
            np.mean([pass_at_k(n, k, 1) for n, k in zip(s.n, s.c)]),
            np.mean([pass_hat_k(n, k, 5) for n, k in zip(s.n, s.c)]),
            np.mean([pass_hat_k(n, k, 10) for n, k in zip(s.n, s.c)])))

print("\n== ReAct vs P&E, same units, seeds 0-4")
d5 = df[df["seed"] <= 4]
pe_units = set(map(tuple, d5[d5["agent"] == "plan_execute"][["task_id", "model"]].drop_duplicates().values))
d5 = d5[[(t, m) in pe_units for t, m in zip(d5["task_id"], d5["model"])]]
d5 = d5[(d5["condition"] == "chain") | ((d5["condition"] == "clean") & d5["task_id"].str.startswith("B"))]
rng = np.random.default_rng(0)
rows = []
for (m, L), s in d5.groupby(["model", "chain_len"]):
    ut = s.groupby(["agent", "task_id"])["success"].agg(n="size", c="sum").reset_index()
    a = ut[ut.agent == "plan_execute"].set_index("task_id")
    b = ut[ut.agent == "react"].set_index("task_id")
    common = sorted(set(a.index) & set(b.index))
    for fam, f, k in (("pass@1", pass_at_k, 1), ("pass^5", pass_hat_k, 5)):
        va = np.array([f(a.n[t], a.c[t], k) for t in common])
        vb = np.array([f(b.n[t], b.c[t], k) for t in common])
        diff = va - vb
        boot = diff[rng.integers(0, len(diff), size=(4000, len(diff)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"model": m, "L": L, "metric": fam, "units": len(common), "react": vb.mean(), "pe": va.mean(),
                     "diff": diff.mean(), "lo": lo, "hi": hi,
                     "pe_eps": int(a.loc[common, "n"].sum()), "pe_succ": int(a.loc[common, "c"].sum())})
print(pd.DataFrame(rows).round(3).to_string(index=False))

"""metrics.py (Member C): log loading, pass@k / pass^k estimators and confidence intervals.

Definitions (n trials per unit = seeds 0..n-1, c = number of successful trials of that unit):
  pass@k  = 1 - C(n-c, k) / C(n, k)   probability that at least one of k runs succeeds   (Chen et al. 2021, unbiased)
  pass^k  =     C(c, k)   / C(n, k)   probability that all k runs succeed                (tau-bench, unbiased)
Both are computed per unit and averaged. A unit is (task_id, paraphrase_idx): paraphrases of one task are never
pooled into one task. For chains the task_id is the chain id ("A05+A01").

Confidence intervals resample CLUSTERS (task_id; all paraphrases of a task stay together), not episodes, because
the 5 seeds of one task are not independent evidence about the agent. With fewer than MIN_CLUSTERS clusters a
percentile bootstrap is unreliable (and collapses to zero width for all-0 / all-1 results), so those rows use a
Wilson interval with n = number of clusters instead; the `ci_method` column says which one was used.

Only the standard library plus numpy/pandas are needed.
"""
import json
import math
from collections import defaultdict
from math import comb

import numpy as np
import pandas as pd

INFRA_PREFIXES = ("api_error", "crash")       # same rule as B's run_e1.is_infra: these rows are retried, never analysed
MIN_CLUSTERS = 8
GROUP_FIELDS = ["agent", "model", "condition", "fault_rate", "chain_len"]
EPISODE_KEY = ("task_id", "agent", "model", "condition", "paraphrase_idx", "fault_rate", "chain_len", "seed")


# ----------------------------------------------------------------------------------------------- loading
def is_infra(err) -> bool:
    return bool(err) and str(err).startswith(INFRA_PREFIXES)


def read_jsonl(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def default_logs(root: str = ".") -> list:
    """Every results/*.jsonl of the project (A's e1, B's e1_b/e2/e3/e4, later runs); duplicates across files are
    resolved by load_logs (same episode key -> newest line wins)."""
    import glob
    import os
    return sorted(glob.glob(os.path.join(root, "results", "*.jsonl")))


def load_logs(paths) -> pd.DataFrame:
    """Read result files, drop infrastructure-error rows, keep the NEWEST row per episode (later file lines win)."""
    rows = []
    for p in paths:
        rows += read_jsonl(p)
    latest = {}
    for r in rows:
        latest[tuple(r.get(k) for k in EPISODE_KEY)] = r
    rows = [r for r in latest.values() if not is_infra(r.get("error", ""))]
    df = pd.DataFrame(rows)
    df["success"] = df["success"].astype(bool)
    df["short_model"] = df["model"].str.split("/").str[-1]
    df["paraphrase_idx"] = df["paraphrase_idx"].astype("float")          # None -> NaN so groupby keeps it
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- estimators
def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n or k < 1:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k. (Written pass_hat_k because '^' is not valid in a Python name.)"""
    if k > n or k < 1:
        return float("nan")
    return comb(c, k) / comb(n, k) if c >= k else 0.0


def wilson(p: float, n: int, z: float = 1.96) -> tuple:
    if n <= 0 or math.isnan(p):
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# ----------------------------------------------------------------------------------------------- units
def unit_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (group, task_id, paraphrase_idx): n trials and c successes. Trials are truncated to the smallest n
    inside each group (seeds 0..n-1) so every unit of a group is scored on the same number of runs."""
    keys = GROUP_FIELDS + ["task_id", "paraphrase_idx"]
    g = df.sort_values("seed").groupby(keys, dropna=False)
    t = g["success"].agg(n="size", c="sum").reset_index()
    nmin = t.groupby(GROUP_FIELDS)["n"].transform("min")
    if (t["n"] != nmin).any():
        parts = []
        for key, sub in df.sort_values("seed").groupby(keys, dropna=False):
            parts.append((key, sub))
        rows = []
        n_group = t.groupby(GROUP_FIELDS)["n"].min().to_dict()
        for key, sub in parts:
            gk = tuple(key[: len(GROUP_FIELDS)])
            nn = n_group[gk]
            rows.append((*key, nn, int(sub["success"].iloc[:nn].sum())))
        t = pd.DataFrame(rows, columns=keys + ["n", "c"])
    t["c"] = t["c"].astype(int)
    return t


def _unit_values(n: int, c: int, ks) -> dict:
    out = {}
    for k in ks:
        out[f"pass@{k}"] = pass_at_k(n, c, k)
        out[f"pass^{k}"] = pass_hat_k(n, c, k)
    return out


def summarize(df: pd.DataFrame, ks=(1, 2, 3, 4, 5), n_boot: int = 4000, seed: int = 0,
              group_fields=None) -> pd.DataFrame:
    """Long table: one row per (group, metric) with estimate and 95% CI. Columns:
    agent, model, condition, fault_rate, chain_len, metric, k, estimate, ci_lo, ci_hi, ci_method, units, clusters, trials.
    The agent is part of the group key: two agent designs are never pooled."""
    group_fields = group_fields or GROUP_FIELDS
    units = unit_table(df)
    rng = np.random.default_rng(seed)
    out = []
    for gkey, sub in units.groupby(group_fields, dropna=False):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        n = int(sub["n"].min())
        # per-unit metric values
        vals = {m: [] for k in ks if k <= n for m in (f"pass@{k}", f"pass^{k}")}
        for _, r in sub.iterrows():
            uv = _unit_values(int(r["n"]), int(r["c"]), [k for k in ks if k <= n])
            for m in vals:
                vals[m].append(uv[m])
        vals = {m: np.array(v, dtype=float) for m, v in vals.items()}
        clusters = sub["task_id"].to_numpy()
        names = sorted(set(clusters))
        idx = {c_: np.where(clusters == c_)[0] for c_ in names}
        nclu = len(names)
        # bootstrap over clusters
        draws = rng.integers(0, nclu, size=(n_boot, nclu))
        for m, v in vals.items():
            csum = np.array([v[idx[c_]].sum() for c_ in names])
            ccnt = np.array([len(idx[c_]) for c_ in names], dtype=float)
            est = v.mean()
            if nclu >= MIN_CLUSTERS:
                boot = csum[draws].sum(axis=1) / ccnt[draws].sum(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                method = "cluster-bootstrap"
                if hi - lo < 1e-12:      # every cluster identical (all 1 / all 0): bootstrap would claim zero uncertainty
                    lo, hi = wilson(est, nclu)
                    method = f"wilson(n={nclu} clusters; bootstrap degenerate)"
            else:
                lo, hi = wilson(est, nclu)
                method = f"wilson(n={nclu} clusters)"
            out.append({**dict(zip(group_fields, gkey)), "metric": m, "k": int(m.split("@")[-1].split("^")[-1]),
                        "estimate": est, "ci_lo": lo, "ci_hi": hi, "ci_method": method,
                        "units": len(sub), "clusters": nclu, "trials": n})
    res = pd.DataFrame(out)
    res["family"] = np.where(res["metric"].str.startswith("pass@"), "pass@k", "pass^k")
    return res.sort_values(group_fields + ["family", "k"]).reset_index(drop=True)


def wide(res: pd.DataFrame, family: str = "pass^k", group_fields=None) -> pd.DataFrame:
    """k as columns with 'est [lo, hi]' strings, handy for printing."""
    group_fields = group_fields or GROUP_FIELDS
    sub = res[res["family"] == family].copy()
    sub["cell"] = sub.apply(lambda r: f"{r['estimate']:.2f} [{r['ci_lo']:.2f},{r['ci_hi']:.2f}]", axis=1)
    return sub.pivot_table(index=group_fields, columns="k", values="cell", aggfunc="first").reset_index()


def paired_difference(df: pd.DataFrame, agent_a: str, agent_b: str, k: int = 1, family: str = "pass@k",
                      n_boot: int = 4000, seed: int = 0) -> pd.DataFrame:
    """agent_a minus agent_b on pass@k or pass^k, per (model, condition), over the units BOTH agents ran.
    The interval resamples tasks (clusters), so it is a paired comparison: each task acts as its own control."""
    units = unit_table(df)
    rng = np.random.default_rng(seed)
    f = pass_at_k if family == "pass@k" else pass_hat_k
    out = []
    keys = ["model", "condition", "fault_rate", "chain_len"]
    for gkey, sub in units.groupby(keys, dropna=False):
        a = sub[sub["agent"] == agent_a].set_index(["task_id", "paraphrase_idx"])
        b = sub[sub["agent"] == agent_b].set_index(["task_id", "paraphrase_idx"])
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        n = int(min(a.loc[common, "n"].min(), b.loc[common, "n"].min()))
        if k > n:
            continue
        diff = np.array([f(int(a.loc[i, "n"]), int(a.loc[i, "c"]), k) - f(int(b.loc[i, "n"]), int(b.loc[i, "c"]), k)
                         for i in common])
        clusters = np.array([i[0] for i in common])
        names = sorted(set(clusters))
        csum = np.array([diff[clusters == c_].sum() for c_ in names])
        ccnt = np.array([(clusters == c_).sum() for c_ in names], dtype=float)
        draws = rng.integers(0, len(names), size=(n_boot, len(names)))
        boot = csum[draws].sum(axis=1) / ccnt[draws].sum(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append({**dict(zip(keys, gkey)), "metric": f"{family.replace('k', str(k))}", "a_minus_b": diff.mean(),
                    "ci_lo": lo, "ci_hi": hi, "units": len(common), "clusters": len(names),
                    "agent_a": agent_a, "agent_b": agent_b})
    return pd.DataFrame(out)

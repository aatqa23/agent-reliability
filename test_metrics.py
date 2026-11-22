"""Offline tests for metrics.py.   python test_metrics.py"""
import itertools
import json
import math
import os
import tempfile

import numpy as np

import metrics as M


def brute(n, c, k, kind):
    """Average over every size-k subset of the n trials (exactly what the closed forms compute)."""
    trials = [1] * c + [0] * (n - c)
    subs = list(itertools.combinations(range(n), k))
    if kind == "at":
        return sum(any(trials[i] for i in s) for s in subs) / len(subs)
    return sum(all(trials[i] for i in s) for s in subs) / len(subs)


def test_estimators_match_brute_force():
    for n in range(1, 8):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                assert math.isclose(M.pass_at_k(n, c, k), brute(n, c, k, "at")), (n, c, k)
                assert math.isclose(M.pass_hat_k(n, c, k), brute(n, c, k, "all")), (n, c, k)


def test_known_values_and_edges():
    assert M.pass_at_k(5, 3, 1) == 0.6 and M.pass_hat_k(5, 3, 1) == 0.6      # k=1 is the plain success rate
    assert M.pass_hat_k(5, 5, 5) == 1.0 and M.pass_hat_k(5, 4, 5) == 0.0
    assert M.pass_at_k(5, 0, 5) == 0.0 and M.pass_at_k(5, 1, 5) == 1.0
    assert math.isnan(M.pass_at_k(5, 3, 6)) and math.isnan(M.pass_hat_k(5, 3, 0))
    assert M.pass_hat_k(5, 4, 2) == 6 / 10                                   # C(4,2)/C(5,2)


def test_wilson():
    lo, hi = M.wilson(0.0, 4)
    assert lo == 0.0 and 0.3 < hi < 0.55                                      # 0 successes of 4 is NOT "exactly zero"
    lo, hi = M.wilson(0.5, 100)
    assert lo < 0.5 < hi and hi - lo < 0.21


def _rows(spec, model="m/x", cond="clean"):
    """spec: {task_id: [bool x n]} -> episode rows."""
    rows = []
    for t, trials in spec.items():
        for s, ok in enumerate(trials):
            rows.append({"task_id": t, "agent": "react", "model": model, "condition": cond, "paraphrase_idx": None,
                         "fault_rate": 0.0, "chain_len": 1, "seed": s, "success": ok, "steps": 3, "tool_calls": 2,
                         "error": "", "latency_s": 1.0})
    return rows


def _write(rows):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def test_load_drops_infra_and_keeps_newest():
    rows = _rows({"T1": [True, True]})
    rows.append({**rows[0], "success": False})                                # same episode again, newer -> wins
    bad = {**rows[1], "seed": 7, "error": "api_error: 429"}
    path = _write(rows + [bad])
    df = M.load_logs([path])
    assert len(df) == 2 and not df[df.seed == 0].success.iloc[0] and 7 not in set(df.seed)


def test_summarize_point_estimates_and_ci_methods():
    spec = {f"T{i:02d}": [True] * 5 for i in range(10)}
    spec["T00"] = [True, True, True, False, False]                             # c=3
    spec["T01"] = [True, True, True, True, False]                              # c=4
    df = M.load_logs([_write(_rows(spec))])
    res = M.summarize(df, n_boot=500)
    g = res.set_index("metric")
    assert math.isclose(g.loc["pass@1", "estimate"], (3 + 4 + 40) / 50)
    assert math.isclose(g.loc["pass^5", "estimate"], 8 / 10)
    assert math.isclose(g.loc["pass^2", "estimate"], (3 / 10 + 6 / 10 + 8) / 10)
    assert g.loc["pass@1", "ci_method"] == "cluster-bootstrap" and g.loc["pass@1", "clusters"] == 10
    assert g.loc["pass@1", "ci_lo"] <= g.loc["pass@1", "estimate"] <= g.loc["pass@1", "ci_hi"]
    small = M.summarize(M.load_logs([_write(_rows({f"S{i}": [False] * 5 for i in range(4)}))]), n_boot=200)
    r = small[small.metric == "pass@1"].iloc[0]
    assert r.estimate == 0.0 and r.ci_method.startswith("wilson") and r.ci_hi > 0.3   # no fake zero-width interval


def test_degenerate_bootstrap_falls_back_to_wilson():
    df = M.load_logs([_write(_rows({f"T{i:02d}": [True] * 5 for i in range(10)}))])
    r = M.summarize(df, n_boot=200)
    r = r[r.metric == "pass@1"].iloc[0]
    assert r.estimate == 1.0 and r.ci_lo < 0.8 and "degenerate" in r.ci_method


def test_paraphrases_are_separate_units_but_one_cluster():
    rows = []
    for p in (0, 1):
        for r in _rows({"T1": [True] * 5 if p == 0 else [False] * 5}):
            rows.append({**r, "condition": "paraphrase", "paraphrase_idx": p})
    df = M.load_logs([_write(rows)])
    u = M.unit_table(df)
    assert len(u) == 2 and u["task_id"].nunique() == 1
    res = M.summarize(df, n_boot=200)
    assert math.isclose(res[res.metric == "pass@1"].estimate.iloc[0], 0.5)


def test_unequal_trials_are_truncated_to_common_n():
    rows = _rows({"A": [True] * 5, "B": [True, False, True]})                   # B only has 3 trials
    df = M.load_logs([_write(rows)])
    u = M.unit_table(df)
    assert set(u["n"]) == {3}
    res = M.summarize(df, n_boot=100)
    assert set(res["k"]) == {1, 2, 3}                                           # no k > common n


def test_agents_are_never_pooled():
    rows = _rows({f"T{i:02d}": [True] * 5 for i in range(10)})
    rows += [{**r, "agent": "plan_execute", "success": False} for r in _rows({f"T{i:02d}": [True] * 5 for i in range(10)})]
    df = M.load_logs([_write(rows)])
    assert len(df) == 100
    res = M.summarize(df, n_boot=100)
    p1 = res[res.metric == "pass@1"].set_index("agent")["estimate"]
    assert p1["react"] == 1.0 and p1["plan_execute"] == 0.0


def test_paired_difference():
    a = _rows({f"T{i:02d}": [True] * 5 for i in range(10)})
    b = [{**r, "agent": "plan_execute"} for r in _rows({f"T{i:02d}": [True, True, True, True, False] if i < 5 else [True] * 5
                                                       for i in range(10)})]
    df = M.load_logs([_write(a + b)])
    d = M.paired_difference(df, "react", "plan_execute", k=1, n_boot=300)
    assert len(d) == 1 and abs(d["a_minus_b"].iloc[0] - 0.1) < 1e-9
    d5 = M.paired_difference(df, "react", "plan_execute", k=5, family="pass^k", n_boot=300)
    assert abs(d5["a_minus_b"].iloc[0] - 0.5) < 1e-9 and d5["ci_lo"].iloc[0] <= 0.5 <= d5["ci_hi"].iloc[0]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")

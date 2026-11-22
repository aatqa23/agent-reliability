"""trace_monitor.py (Member C): trace-based failure monitor (episode level) = predict from the agent's TRACE alone whether an episode failed,
i.e. a checker-free failure monitor. Everything is split BY TASK: a task (or chain) used for training never appears in
the test data.

Features (no goal-state checker, no latency [it is a cache artefact: cached replays take ~0 s], no error text):
  trace    : steps, tool_calls, calls_per_step, steps_per_msg, calls_per_msg   (per message = divided by chain length)
  context  : is_120b, chain_len, fault_rate, is_paraphrase
Two feature sets are compared: "context" only (what the run setting alone tells you) vs "context+trace".

Evaluation
  1. leave-one-task-out CV (every episode is scored by a model that never saw its task) -> the numbers to trust
  2. one fixed train/test split by task (seeded) -> the saved model and the "test scores" for the report
Scores: ROC-AUC and average precision for detecting FAILURES, plus precision/recall at 0.5. 95% intervals bootstrap the
tasks. Reported for all episodes, single-task episodes only (the hard case: failures are rare) and chains only.

  python trace_monitor.py          (run from the project root; pools results/*.jsonl)
"""
import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import metrics as M

CONTEXT = ["is_120b", "chain_len", "fault_rate", "is_paraphrase", "is_plan_execute"]
TRACE = ["steps", "tool_calls", "calls_per_step", "steps_per_msg", "calls_per_msg"]
FEATURE_SETS = {"context": CONTEXT, "context+trace": CONTEXT + TRACE}


def build(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["failure"] = (~d["success"]).astype(int)
    d["is_120b"] = d["short_model"].eq("gpt-oss-120b").astype(int)
    d["is_paraphrase"] = d["condition"].eq("paraphrase").astype(int)
    d["is_plan_execute"] = d["agent"].ne("react").astype(int)
    d["calls_per_step"] = d["tool_calls"] / d["steps"].clip(lower=1)
    d["steps_per_msg"] = d["steps"] / d["chain_len"]
    d["calls_per_msg"] = d["tool_calls"] / d["chain_len"]
    d["group"] = d["task_id"]
    d["kind"] = np.where(d["condition"].eq("chain"), "chains", "single")
    return d.reset_index(drop=True)


def make_model(name: str):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    if name == "gboost":
        return GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.1, subsample=0.8, random_state=0)
    raise ValueError(name)


def score_block(y, p, thr=0.5) -> dict:
    out = {"n": int(len(y)), "failures": int(y.sum())}
    if len(set(y)) < 2:
        return {**out, "roc_auc": np.nan, "avg_precision": np.nan, "precision@0.5": np.nan, "recall@0.5": np.nan}
    pred = (p >= thr).astype(int)
    return {**out, "roc_auc": roc_auc_score(y, p), "avg_precision": average_precision_score(y, p),
            "precision@0.5": precision_score(y, pred, zero_division=0), "recall@0.5": recall_score(y, pred, zero_division=0)}


def boot_ci(d: pd.DataFrame, pcol: str, metric: str, n_boot=500, seed=0):
    """Bootstrap over tasks (groups): resample groups, recompute the metric on the pooled predictions."""
    rng = np.random.default_rng(seed)
    ys = [g["failure"].to_numpy() for _, g in d.groupby("group")]
    ps = [g[pcol].to_numpy() for _, g in d.groupby("group")]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ys), size=len(ys))
        y = np.concatenate([ys[i] for i in pick])
        if y.min() == y.max():
            continue
        p = np.concatenate([ps[i] for i in pick])
        vals.append(roc_auc_score(y, p))     # metric is ROC-AUC (the only one bootstrapped)
    return (np.nan, np.nan) if len(vals) < 50 else tuple(np.percentile(vals, [2.5, 97.5]))


def logo_predictions(d, feats, model_name):
    """Out-of-fold failure probabilities, leaving one task (group) out at a time."""
    p = np.zeros(len(d))
    X, y = d[feats].to_numpy(float), d["failure"].to_numpy()
    for g in d["group"].unique():
        te = (d["group"] == g).to_numpy()
        m = make_model(model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[~te], y[~te])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def fixed_split(d, test_frac=0.3, min_fail=8, max_tries=500):
    """Seeded split of TASKS into train/test; the first seed whose test AND train parts each hold enough failures,
    successes and both episode kinds (so the test score is not undefined or trivial)."""
    groups = sorted(d["group"].unique())
    for seed in range(max_tries):
        rng = np.random.default_rng(seed)
        test = {str(g) for g in rng.choice(groups, size=max(2, round(len(groups) * test_frac)), replace=False)}
        te = d["group"].isin(test)
        tc = d[te & (d["kind"] == "chains")]
        ok = (d[te]["failure"].sum() >= min_fail and d[~te]["failure"].sum() >= min_fail and
              (1 - d[te]["failure"]).sum() >= min_fail and
              d[te & (d["kind"] == "single")]["failure"].sum() >= 3 and
              tc["group"].nunique() >= 2 and tc["failure"].nunique() == 2)   # chain test part must hold both outcomes
        if ok:
            return seed, sorted(test)
    raise RuntimeError("no acceptable split found")


def fig_classifier(t, figdir, note=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    plt.rcParams.update({"figure.facecolor": surf, "axes.facecolor": surf, "savefig.facecolor": surf, "axes.edgecolor": grid,
                         "axes.labelcolor": ink2, "text.color": ink, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.grid": True, "grid.color": grid, "axes.axisbelow": True, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "axes.titleweight": "bold",
                         "axes.titlelocation": "left"})
    cv = t[t["cv"] == "leave-one-task-out"]
    subsets = ["all", "single", "chains"]
    names = {"all": "All episodes", "single": "Single-task episodes", "chains": "Chains"}
    combos = [("logreg", "context", "#8a8985"), ("logreg", "context+trace", "#2a78d6"),
              ("gboost", "context", "#c9c8c3"), ("gboost", "context+trace", "#eb6834")]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    w = 0.19
    for i, (m, f, col) in enumerate(combos):
        xs, ys, lo, hi = [], [], [], []
        for j, sset in enumerate(subsets):
            r = cv[(cv.model == m) & (cv.features == f) & (cv.subset == sset)].iloc[0]
            xs.append(j + (i - 1.5) * w)
            ys.append(r.roc_auc)
            lo.append(r.roc_auc - r.auc_lo if r.auc_lo == r.auc_lo else 0)
            hi.append(r.auc_hi - r.roc_auc if r.auc_hi == r.auc_hi else 0)
        ax.bar(xs, ys, w * 0.88, color=col, label=f"{m}, {f}")
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=ink2, elinewidth=1, capsize=2)
    ax.axhline(0.5, color=ink2, lw=1, ls="--")
    ax.text(2.45, 0.505, "chance", ha="right", va="bottom", fontsize=8, color=ink2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([names[s] for s in subsets])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("ROC-AUC for detecting failures")
    ax.set_title("Leave-one-task-out: trace features help on chains, not on single tasks", loc="left")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.01, 0.005, "95% intervals bootstrap tasks/chains. " + note,
             fontsize=7.5, color=ink2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(figdir, "fig6_trace_monitor_auc.png"), dpi=170)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", default=None, help="default: every results/*.jsonl")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--figs", default="analysis/figures")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.figs, exist_ok=True)
    d = build(M.load_logs(a.logs or M.default_logs()))
    print(f"{len(d)} episodes, {d['failure'].sum()} failures ({d['failure'].mean():.1%}), {d['group'].nunique()} tasks/chains")
    print(d.groupby("kind")["failure"].agg(["size", "sum"]).rename(columns={"size": "episodes", "sum": "failures"}))

    # ---- 1. leave-one-task-out
    rows = []
    for model_name in ("logreg", "gboost"):
        for fs_name, feats in FEATURE_SETS.items():
            d[f"p_{model_name}_{fs_name}"] = logo_predictions(d, feats, model_name)
    for model_name in ("logreg", "gboost"):
        for fs_name in FEATURE_SETS:
            pcol = f"p_{model_name}_{fs_name}"
            for subset in ("all", "single", "chains"):
                s = d if subset == "all" else d[d["kind"] == subset]
                sc = score_block(s["failure"].to_numpy(), s[pcol].to_numpy())
                lo, hi = boot_ci(s, pcol, "roc_auc") if subset == "all" or s["group"].nunique() >= 8 else (np.nan, np.nan)
                rows.append({"cv": "leave-one-task-out", "model": model_name, "features": fs_name, "subset": subset,
                             **sc, "auc_lo": lo, "auc_hi": hi})
    # ---- 2. fixed split by task
    seed, test_groups = fixed_split(d)
    te = d["group"].isin(test_groups).to_numpy()
    print(f"\nfixed split (seed {seed}): {len(test_groups)} test tasks: {test_groups}")
    saved = {}
    for model_name in ("logreg", "gboost"):
        for fs_name, feats in FEATURE_SETS.items():
            m = make_model(model_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(d.loc[~te, feats].to_numpy(float), d.loc[~te, "failure"].to_numpy())
            p = m.predict_proba(d.loc[te, feats].to_numpy(float))[:, 1]
            dt = d[te].copy()
            dt["p"] = p
            for subset in ("all", "single", "chains"):
                s = dt if subset == "all" else dt[dt["kind"] == subset]
                sc = score_block(s["failure"].to_numpy(), s["p"].to_numpy())
                lo, hi = boot_ci(s, "p", "roc_auc") if s["group"].nunique() >= 3 else (np.nan, np.nan)
                rows.append({"cv": f"fixed-split(seed={seed})", "model": model_name, "features": fs_name, "subset": subset,
                             **sc, "auc_lo": lo, "auc_hi": hi})
            saved[(model_name, fs_name)] = m
    t = pd.DataFrame(rows)
    t.insert(0, "train_tasks", [None] * len(t))
    t = t.drop(columns="train_tasks")
    num = t.select_dtypes("number").columns
    t[num] = t[num].round(3)
    t.to_csv(os.path.join(a.out, "trace_monitor_scores.csv"), index=False)

    # saved artefact: the fixed-split logistic model on context+trace (trained on train tasks only)
    best = ("logreg", "context+trace")
    feats = FEATURE_SETS[best[1]]
    joblib.dump({"model": saved[best], "features": feats, "test_tasks": test_groups, "split_seed": seed,
                 "note": "predicts P(episode failed) from trace features; trained on train tasks only"},
                os.path.join(a.out, "trace_monitor.joblib"))
    lr = saved[best]
    coefs = {k: float(v) for k, v in zip(feats, lr[-1].coef_[0].round(3))}
    json.dump({"split_seed": seed, "test_tasks": test_groups,
               "train_tasks": sorted(set(d["group"]) - set(test_groups)),
               "logreg_standardized_coefficients(+ = more likely failure)": coefs},
              open(os.path.join(a.out, "trace_monitor_split_and_coefs.json"), "w"), indent=1)
    d[["agent", "task_id", "model", "condition", "fault_rate", "chain_len", "seed", "success", "steps", "tool_calls",
       "p_logreg_context", "p_logreg_context+trace", "p_gboost_context", "p_gboost_context+trace"]].to_csv(
        os.path.join(a.out, "trace_monitor_loto_predictions.csv"), index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n", t.to_string(index=False))
    k = d.groupby("kind")["failure"].agg(["size", "sum"])
    fig_classifier(t, a.figs, f"Single-task episodes: {int(k.loc['single', 'sum'])} failures of {int(k.loc['single', 'size'])}; chain episodes: {int(k.loc['chains', 'sum'])} of {int(k.loc['chains', 'size'])} ({d[d['kind'] == 'chains']['group'].nunique()} chains).")
    print("\nstandardized logreg coefficients:", coefs)
    return t, d


if __name__ == "__main__":
    main()

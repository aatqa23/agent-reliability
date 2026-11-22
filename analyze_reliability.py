"""analyze_reliability.py (Member C): pass@k / pass^k tables, E5 run-budget analysis, chain-decay fit, figures.

  python analyze_reliability.py                       # pools every results/*.jsonl of the project (run from the root)
  python analyze_reliability.py --logs results/e1.jsonl results/e2.jsonl ...

Outputs -> analysis/*.csv and analysis/figures/*.png.  Agents are never pooled: with two agents in the logs every
table has an agent column, figures are drawn per agent, and analysis/agent_comparison.csv holds paired differences.
"""
import argparse
import itertools
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

import metrics as M

# validated reference palette (dataviz skill, light mode): blue and orange are reserved for the two MODELS;
# conditions get their own colours.
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
MODEL_COL = {"gpt-oss-120b": "#2a78d6", "gpt-oss-20b": "#eb6834"}
COND_COL = {"clean": "#52514e", "paraphrase": "#1baf7a", "fault 0.1": "#e87ba4", "fault 0.2": "#6250d6",
            "chain L2": "#eda100", "chain L3": "#008300", "chain L4": "#e34948"}
COND_ORDER = ["clean", "paraphrase", "fault 0.1", "fault 0.2", "chain L2", "chain L3", "chain L4"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
    })


def cond_label(r) -> str:
    if r["condition"] == "fault":
        return f"fault {r['fault_rate']:g}"
    if r["condition"] == "chain":
        return f"chain L{int(r['chain_len'])}"
    return r["condition"]


def order_conds(conds):
    known = [c for c in COND_ORDER if c in set(conds)]
    return known + sorted(set(conds) - set(known))


def tag(agents, agent):
    """File-name suffix: empty with a single agent, '_<agent>' otherwise."""
    return "" if len(agents) == 1 else f"_{agent}"


# ------------------------------------------------------------------------------------------------ tables
def table_main(df, out):
    res = M.summarize(df)
    res["cond"] = res.apply(lambda r: cond_label(r), axis=1)
    res["model"] = res["model"].str.split("/").str[-1]
    res.to_csv(os.path.join(out, "passk_long.csv"), index=False)
    rows = []
    for (a, m, c), s in res[res["k"].isin([1, 5])].groupby(["agent", "model", "cond"]):
        row = {"agent": a, "model": m, "cond": c, "clusters": int(s["clusters"].iloc[0]), "units": int(s["units"].iloc[0]),
               "trials": int(s["trials"].iloc[0])}
        for _, r in s.iterrows():
            if r["metric"] == "pass^1":
                continue
            row[r["metric"]] = round(r["estimate"], 3)
            row[r["metric"] + "_lo"], row[r["metric"] + "_hi"] = round(r["ci_lo"], 3), round(r["ci_hi"], 3)
        row["ci_method"] = s["ci_method"].iloc[0]
        rows.append(row)
    t = pd.DataFrame(rows)
    t["cond"] = pd.Categorical(t["cond"], order_conds(t["cond"]), ordered=True)
    t = t.sort_values(["agent", "model", "cond"]).reset_index(drop=True)
    t["gap_pass1_minus_passhat5"] = (t["pass@1"] - t["pass^5"]).round(3)
    t.to_csv(os.path.join(out, "summary_table.csv"), index=False)
    return res, t


def agent_comparison(df, out):
    agents = sorted(df["agent"].unique())
    if len(agents) < 2:
        return pd.DataFrame()
    parts = []
    for a, b in itertools.combinations(agents, 2):
        for k, fam in ((1, "pass@k"), (5, "pass^k")):
            parts.append(M.paired_difference(df, a, b, k=k, family=fam))
    t = pd.concat([p for p in parts if len(p)], ignore_index=True) if parts else pd.DataFrame()
    if len(t):
        t["model"] = t["model"].str.split("/").str[-1]
        t = t.round(4)
        t.to_csv(os.path.join(out, "agent_comparison.csv"), index=False)
    return t


# ------------------------------------------------------------------------------------------------ E5
def unit_matrix(df):
    """{(agent, model, cond): units x seeds bool matrix}, only the seeds present for every unit of the cell."""
    d = df.copy()
    d["cond"] = d.apply(cond_label, axis=1)
    out = {}
    for (a, m, c), sub in d.groupby(["agent", "short_model", "cond"]):
        piv = sub.pivot_table(index=["task_id", "paraphrase_idx"], columns="seed", values="success", aggfunc="first",
                              dropna=False).dropna(axis=1)
        out[(a, m, c)] = piv.to_numpy().astype(bool)
    return out


def e5(df, out):
    """E5, run budget: what if only n' of the available seeds per task were affordable? Every subset of n' seeds is
    scored (n' = 1 .. number of seeds; the shared plan has 5 seeds, so the range of "2 to 10 trials" can only be
    covered up to 5 here).
      budget   pass@1 estimate over subsets (mean/sd/min/max) and pass^n' (all n' runs succeed)
      flips    how often the 120b-vs-20b ordering of pass@1 is reversed or tied (vs the all-seed ordering)
      ranking  Kendall tau-b between the ranking of ALL (agent, model, condition) cells by pass@1 from n' runs and
               the ranking from all seeds; 'stable at' = the smallest n' whose WORST subset has tau >= 0.9."""
    mats = unit_matrix(df)
    rows = []
    for (a, m, c), A in sorted(mats.items()):
        ns = A.shape[1]
        for n in range(1, ns + 1):
            p1, pk = [], []
            for S in itertools.combinations(range(ns), n):
                sub = A[:, list(S)]
                p1.append(sub.mean())
                pk.append(sub.all(axis=1).mean())
            rows.append({"agent": a, "model": m, "cond": c, "n_runs": n, "subsets": len(p1), "pass@1_mean": np.mean(p1),
                         "pass@1_sd": np.std(p1), "pass@1_min": np.min(p1), "pass@1_max": np.max(p1),
                         "pass^n_mean": np.mean(pk)})
    budget = pd.DataFrame(rows).round(4)
    budget.to_csv(os.path.join(out, "e5_run_budget.csv"), index=False)

    flips = []
    for agent in sorted({a for a, _, _ in mats}):
        models = sorted({m for a, m, _ in mats if a == agent})
        if len(models) != 2:
            continue
        x, y = models
        for c in order_conds({c for a, _, c in mats if a == agent}):
            if (agent, x, c) not in mats or (agent, y, c) not in mats:
                continue
            A, B = mats[(agent, x, c)], mats[(agent, y, c)]
            ns = min(A.shape[1], B.shape[1])
            full = np.sign(round(A[:, :ns].mean() - B[:, :ns].mean(), 9))
            if full == 0:
                continue
            for n in range(1, ns + 1):
                diffs = [np.sign(round(A[:, list(S)].mean() - B[:, list(S)].mean(), 9))
                         for S in itertools.combinations(range(ns), n)]
                flips.append({"agent": agent, "cond": c, "n_runs": n, "better_at_full": x if full > 0 else y,
                              "subsets": len(diffs), "reversed_or_tied_rate": float(np.mean([d != full for d in diffs]))})
    flips = pd.DataFrame(flips).round(4)
    flips.to_csv(os.path.join(out, "e5_ranking_flips.csv"), index=False)

    ns = max(A.shape[1] for A in mats.values())
    cells = sorted(c for c in mats if mats[c].shape[1] == ns)       # only cells with ALL seeds: incomplete cells would cap n'
    full_p = np.array([mats[c][:, :ns].mean() for c in cells])
    rank_rows = []
    if len(cells) >= 3:
        for n in range(1, ns + 1):
            taus = []
            for S in itertools.combinations(range(ns), n):
                p = np.array([mats[c][:, list(S)].mean() for c in cells])
                taus.append(stats.kendalltau(p, full_p).statistic)
            taus = np.array(taus, dtype=float)
            rank_rows.append({"n_runs": n, "subsets": len(taus), "cells": len(cells), "tau_mean": np.nanmean(taus),
                              "tau_min": np.nanmin(taus), "tau_p10": np.nanpercentile(taus, 10)})
    ranking = pd.DataFrame(rank_rows).round(4)
    ranking.to_csv(os.path.join(out, "e5_ranking_stability.csv"), index=False)
    stable = ranking[ranking["tau_min"] >= 0.9]["n_runs"].min() if len(ranking) else np.nan
    return budget, flips, ranking, stable


# ------------------------------------------------------------------------------------------------ chain decay
def chain_decay(df, out):
    """Chain decay: predicted success of an L-message chain vs measured success.
      pooled   p**L, with p = single-task pass@1 pooled over all CLEAN runs (A's and B's tasks): no fitted parameter
      members  product of the chain's own member tasks' clean pass@1 (a per-chain version of the same prediction)
      fitted   s**L, s fitted by binomial maximum likelihood on clean singles (L=1) and 2-message chains
    s_L = observed**(1/L) is the implied per-message success; under independent errors it would not depend on L.
    Intervals: Wilson with n = number of independent tasks/chains."""
    d = df.copy()
    rows = []
    for (agent, model), sub in d.groupby(["agent", "short_model"]):
        cl, ch = sub[sub["condition"] == "clean"], sub[sub["condition"] == "chain"]
        if cl.empty or ch.empty:
            continue
        p_task = cl.groupby("task_id")["success"].mean()
        n1, c1, k1 = len(cl), int(cl["success"].sum()), cl["task_id"].nunique()
        p_pool = c1 / n1
        per = {L: (len(s), int(s["success"].sum()), s["task_id"].nunique()) for L, s in ch.groupby("chain_len")}
        short = {1: (n1, c1, k1), **({2: per[2]} if 2 in per else {})}

        def nll(s):
            ll = 0.0
            for L, (n, c, _) in short.items():
                p = min(max(s ** L, 1e-9), 1 - 1e-9)
                ll += c * math.log(p) + (n - c) * math.log(1 - p)
            return -ll
        s_hat = float(optimize.minimize_scalar(nll, bounds=(1e-6, 1 - 1e-9), method="bounded").x) if len(short) > 1 else p_pool
        rows.append({"agent": agent, "model": model, "L": 1, "episodes": n1, "clusters": k1, "successes": c1,
                     "observed": round(p_pool, 3), "ci_lo": round(M.wilson(p_pool, k1)[0], 3),
                     "ci_hi": round(M.wilson(p_pool, k1)[1], 3), "pred_pooled_p^L": round(p_pool, 3),
                     "pred_member_product": round(p_pool, 3), "pred_fitted_s^L": round(s_hat, 3), "s_fit": round(s_hat, 3),
                     "implied_s_L": round(p_pool, 3), "implied_s_lo": round(M.wilson(p_pool, k1)[0], 3),
                     "implied_s_hi": round(M.wilson(p_pool, k1)[1], 3), "binom_P(X<=obs|member_product)": None})
        for L, s in sorted(ch.groupby("chain_len")):
            n, c, ncl = per[L]
            obs = c / n
            lo, hi = M.wilson(obs, ncl)
            members = []
            for cid in s["task_id"].unique():
                members.append(float(np.prod([p_task.get(m, p_pool) for m in cid.split("+")])))
            pred_members = float(np.mean(members))
            pval = float(stats.binom.cdf(c, n, pred_members))
            rows.append({"agent": agent, "model": model, "L": int(L), "episodes": n, "clusters": ncl, "successes": c,
                         "observed": round(obs, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                         "pred_pooled_p^L": round(p_pool ** L, 3), "pred_member_product": round(pred_members, 3),
                         "pred_fitted_s^L": round(s_hat ** L, 3), "s_fit": round(s_hat, 3),
                         "implied_s_L": round(obs ** (1 / L), 3), "implied_s_lo": round(lo ** (1 / L), 3),
                         "implied_s_hi": round(hi ** (1 / L), 3), "binom_P(X<=obs|member_product)": float(f"{max(pval, 1e-300):.2e}")})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "chain_decay.csv"), index=False)
    return t


# ------------------------------------------------------------------------------------------------ figures
def _save(fig, figdir, name):
    fig.savefig(os.path.join(figdir, name), dpi=170)
    plt.close(fig)


def fig_passk(res, figdir, agent, suffix):
    res = res[res["agent"] == agent]
    models = sorted(res["model"].unique())
    conds = order_conds(res["cond"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(10.5, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        for c in conds:
            s = res[(res.model == m) & (res.cond == c) & (res.family == "pass^k")].sort_values("k")
            if s.empty:
                continue
            ax.plot(s["k"], s["estimate"], color=COND_COL.get(c, "#8a8985"), marker=MARKERS[COND_ORDER.index(c) % 7],
                    ms=6, mew=1.5, mec=SURFACE, label=c)
        ax.set_title(m, loc="left")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("k (runs that must ALL succeed)")
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("pass^k")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle(f"pass^k vs k ({agent}): reliability decays under faults and long chains",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.005, "clean = all 30 tasks; paraphrase/fault = B tasks; chains mix A and B tasks (4 per length). "
             "Intervals in analysis/passk_long.csv.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    _save(fig, figdir, f"fig1_passk_curves{suffix}.png")


def fig_summary(t, figdir, agent, suffix):
    t = t[t["agent"] == agent]
    conds = order_conds(t["cond"].astype(str).unique())
    models = sorted(t["model"].unique())
    fig, ax = plt.subplots(figsize=(10, 4.3))
    w = 0.19
    x = np.arange(len(conds))
    k = 0
    for m in models:
        for kind, hatch in (("pass@1", None), ("pass^5", "////")):
            vals, lo, hi = [], [], []
            for c in conds:
                r = t[(t.model == m) & (t["cond"].astype(str) == c)]
                vals.append(r[kind].iloc[0] if len(r) else np.nan)
                lo.append(r[kind + "_lo"].iloc[0] if len(r) else np.nan)
                hi.append(r[kind + "_hi"].iloc[0] if len(r) else np.nan)
            vals, lo, hi = map(np.array, (vals, lo, hi))
            pos = x + (k - 1.5) * w
            ax.bar(pos, vals, w * 0.86, color=MODEL_COL.get(m, "#8a8985"), hatch=hatch, edgecolor="white", linewidth=0.0,
                   label=f"{m}  {kind}")
            ax.errorbar(pos, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
            k += 1
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.legend(loc="upper right", ncol=2, fontsize=8.5)
    ax.set_title(f"pass@1 (solid) vs pass^5 (hatched), 95% intervals ({agent})", loc="left")
    fig.text(0.01, 0.005, "Intervals: cluster bootstrap over tasks (>=8 tasks) or Wilson on #tasks/chains (fewer, or all-equal).",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, figdir, f"fig2_pass1_vs_pass5{suffix}.png")


def fig_e5(budget, flips, ranking, stable, figdir, agent, suffix):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    cond = "fault 0.2"
    for m in sorted(budget["model"].unique()):
        s = budget[(budget.agent == agent) & (budget.model == m) & (budget["cond"] == cond)].sort_values("n_runs")
        if s.empty:
            continue
        ax.fill_between(s["n_runs"], s["pass@1_min"], s["pass@1_max"], color=MODEL_COL.get(m, "#8a8985"), alpha=0.18, lw=0)
        ax.plot(s["n_runs"], s["pass@1_mean"], color=MODEL_COL.get(m, "#8a8985"), marker="o", ms=6, mec=SURFACE, mew=1.5,
                label=m)
    ax.set_title(f"A. pass@1 over seed subsets ({cond})", loc="left")
    ax.set_xlabel("runs per task used")
    ax.set_ylabel("pass@1 estimate")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(loc="lower right", fontsize=8.5)
    ax = axes[1]
    fl = flips[flips["agent"] == agent] if len(flips) else flips
    for c in order_conds(fl["cond"].unique()) if len(fl) else []:
        s = fl[fl["cond"] == c].sort_values("n_runs")
        ax.plot(s["n_runs"], s["reversed_or_tied_rate"], color=COND_COL.get(c, "#8a8985"),
                marker=MARKERS[COND_ORDER.index(c) % 7] if c in COND_ORDER else "o", ms=6, mec=SURFACE, mew=1.5, label=c)
    ax.set_title("B. 120b-vs-20b ordering reversed or tied", loc="left")
    ax.set_xlabel("runs per task used")
    ax.set_ylabel("share of subsets")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(-0.03, 1.0)
    if len(fl):
        ax.legend(fontsize=8, loc="upper right", title="condition", title_fontsize=8)
    ax = axes[2]
    if len(ranking):
        ax.fill_between(ranking["n_runs"], ranking["tau_min"], ranking["tau_mean"], color="#52514e", alpha=0.15, lw=0)
        ax.plot(ranking["n_runs"], ranking["tau_mean"], color="#52514e", marker="o", ms=6, mec=SURFACE, mew=1.5, label="mean")
        ax.plot(ranking["n_runs"], ranking["tau_min"], color="#e34948", marker="v", ms=6, mec=SURFACE, mew=1.5, label="worst subset")
        ax.axhline(0.9, color=INK2, ls="--", lw=1)
        ax.text(1.05, 0.915, "tau 0.9", fontsize=8, color=INK2)
        ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("C. Ranking of all cells vs full-data ranking", loc="left")
    ax.set_xlabel("runs per task used")
    ax.set_ylabel("Kendall tau-b")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(-0.05, 1.05)
    msg = "stable (worst subset tau >= 0.9) from " + (f"{int(stable)} runs" if stable == stable else "never within the runs available")
    fig.suptitle(f"E5 run budget ({agent}): few runs per task can reverse conclusions; ranking {msg}", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.005, "Every subset of n runs of the available seeds is scored. Panel B omits conditions where the full-data "
             "scores of the two models are equal (clean).", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    _save(fig, figdir, f"fig3_e5_run_budget{suffix}.png")


def fig_chain(t, figdir, agent, suffix):
    t = t[t["agent"] == agent]
    if t.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for m in sorted(t["model"].unique()):
        s = t[t.model == m].sort_values("L")
        col = MODEL_COL.get(m, "#8a8985")
        axes[0].errorbar(s["L"], s["observed"], yerr=[s["observed"] - s["ci_lo"], s["ci_hi"] - s["observed"]], color=col,
                         marker="o", ms=6, mec=SURFACE, mew=1.5, capsize=3, elinewidth=1, label=f"{m} observed")
        axes[0].plot(s["L"], s["pred_pooled_p^L"], color=col, ls="--", lw=1.5, label=f"{m} predicted p^L (p = clean pass@1)")
        axes[1].errorbar(s["L"], s["implied_s_L"], yerr=[s["implied_s_L"] - s["implied_s_lo"], s["implied_s_hi"] - s["implied_s_L"]],
                         color=col, marker="o", ms=6, mec=SURFACE, mew=1.5, capsize=3, elinewidth=1, label=m)
    axes[0].set_title("A. Chain success: measured vs predicted", loc="left")
    axes[0].set_ylabel("pass@1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 0.035))
    axes[1].set_title("B. Implied per-message success  observed^(1/L)", loc="left")
    axes[1].set_ylabel("per-message success")
    for ax in axes:
        ax.set_xticks(sorted(t["L"].unique()))
        ax.set_xlabel("messages in the chain (L)")
        ax.set_ylim(-0.03, 1.05)
    fig.suptitle(f"Chains fail far faster than independent per-message errors predict ({agent})", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.005, "L=1: clean single tasks (all 30). Bars: Wilson, n = number of tasks/chains. "
             "Per-chain and fitted predictions: analysis/chain_decay.csv.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    _save(fig, figdir, f"fig4_chain_decay{suffix}.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", default=None, help="default: every results/*.jsonl")
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args(argv)
    logs = a.logs or M.default_logs()
    if not logs:
        raise SystemExit("no logs found: run from the project root (results/*.jsonl) or pass --logs")
    figdir = os.path.join(a.out, "figures")
    os.makedirs(figdir, exist_ok=True)
    style()
    df = M.load_logs(logs)
    agents = sorted(df["agent"].unique())
    print(f"{len(df)} episodes from {len(logs)} files; agents: {agents}")
    res, t = table_main(df, a.out)
    budget, flips, ranking, stable = e5(df, a.out)
    decay = chain_decay(df, a.out)
    cmp_ = agent_comparison(df, a.out)
    for agent in agents:
        sfx = tag(agents, agent)
        fig_passk(res, figdir, agent, sfx)
        fig_summary(t, figdir, agent, sfx)
        fig_e5(budget, flips, ranking, stable, figdir, agent, sfx)
        fig_chain(decay, figdir, agent, sfx)
    pd.set_option("display.width", 230, "display.max_columns", 40)
    print("\n== summary_table.csv\n", t.drop(columns=["units", "ci_method"]).to_string(index=False))
    print("\n== e5_ranking_stability.csv (stable at:", stable, ")\n", ranking.to_string(index=False))
    print("\n== e5_ranking_flips.csv (n_runs 1 and 3)\n", flips[flips.n_runs.isin([1, 3])].to_string(index=False))
    print("\n== chain_decay.csv\n", decay.drop(columns=["successes", "clusters", "implied_s_lo", "implied_s_hi"]).to_string(index=False))
    if len(cmp_):
        print("\n== agent_comparison.csv\n", cmp_.to_string(index=False))


if __name__ == "__main__":
    main()

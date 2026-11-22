"""Figures 2-7 of the paper, drawn from the analysis CSVs and the logs (no API calls).

  python analyze_reliability.py && python task_classifier.py && python trace_monitor.py && python paper_figures.py

Writes paper/fig2_bars.png ... paper/fig7_detector.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metrics as M
from analyze_reliability import COND_COL, COND_ORDER, GRID, INK2, MARKERS, MODEL_COL, SURFACE, cond_label, style

OUT = "paper"
os.makedirs(OUT, exist_ok=True)
style()
plt.rcParams.update({"font.family": "Inter", "font.size": 10})
PE_COL = {"gpt-oss-120b": "#2a78d6", "gpt-oss-20b": "#eb6834"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight")
    plt.close(fig)


S = pd.read_csv("analysis/summary_table.csv")
S["model"] = S["model"].str.replace("openai/", "", regex=False)
R = S[S.agent == "react"]

# ---------------- Fig 2: pass@1 vs pass^5 bars (ReAct)
fig, ax = plt.subplots(figsize=(7, 2.8))
conds = [c for c in COND_ORDER if c in set(R.cond)]
x = np.arange(len(conds))
w = 0.2
for i, m in enumerate(["gpt-oss-120b", "gpt-oss-20b"]):
    for j, (kind, hatch) in enumerate((("pass@1", None), ("pass^5", "////"))):
        sub = R[R.model == m].set_index("cond").reindex(conds)
        pos = x + (2 * i + j - 1.5) * w
        est, lo, hi = sub[kind], sub[f"{kind}_lo"], sub[f"{kind}_hi"]
        ax.bar(pos, est, w * 0.92, color=MODEL_COL[m], hatch=hatch, edgecolor=SURFACE if hatch else None, lw=0,
               label=f"{m} {kind}")
        ax.errorbar(pos, est, yerr=[est - lo, hi - est], fmt="none", ecolor="#4a4a48", elinewidth=1, capsize=2)
ax.set_xticks(x)
ax.set_xticklabels(conds)
ax.set_ylabel("score")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
save(fig, "fig2_bars.png")

# ---------------- Fig 3: pass^k for k = 1..10 (ReAct), from the logs
df = M.load_logs(M.default_logs("."))
df["model"] = df["model"].str.replace("openai/", "", regex=False)
df = df[df.agent == "react"]
df["cond"] = df.apply(cond_label, axis=1)
fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
for ax, m in zip(axes, ["gpt-oss-120b", "gpt-oss-20b"]):
    for ci, c in enumerate(conds):
        sub = df[(df.model == m) & (df.cond == c)]
        ut = sub.groupby(["task_id", "paraphrase_idx"], dropna=False)["success"].agg(n="size", c="sum")
        n = int(ut["n"].min())
        ks = list(range(1, n + 1))
        ys = [np.mean([M.pass_hat_k(int(a), int(b), k) for a, b in zip(ut["n"], ut["c"])]) for k in ks]
        ax.plot(ks, ys, color=COND_COL[c], marker=MARKERS[ci], ms=4, mec=SURFACE, mew=1, lw=1.6, label=c)
    ax.set_title(m)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("k (runs that must all succeed)")
    ax.set_ylim(-0.03, 1.03)
axes[0].set_ylabel("pass^k")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, fontsize=7.5, bbox_to_anchor=(0.5, -0.2), columnspacing=1.0, handlelength=1.6)
save(fig, "fig3_passk.png")

# ---------------- Fig 4: chain decay, ReAct (solid) and plan-then-execute (dotted)
C = pd.read_csv("analysis/chain_decay.csv")
C["model"] = C["model"].str.replace("openai/", "", regex=False)
fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
for m in ["gpt-oss-120b", "gpt-oss-20b"]:
    col = MODEL_COL[m]
    r = C[(C.agent == "react") & (C.model == m)].sort_values("L")
    axes[0].errorbar(r.L, r.observed, yerr=[r.observed - r.ci_lo, r.ci_hi - r.observed], color=col, marker="o", ms=5,
                     mec=SURFACE, capsize=2, elinewidth=1, label=f"{m} ReAct")
    axes[0].plot(r.L, r["pred_pooled_p^L"], color=col, ls="--", lw=1.3, label=f"{m} predicted p^L")
    axes[1].plot(r.L, r.implied_s_L, color=col, marker="o", ms=5, mec=SURFACE, label=f"{m} ReAct")
    p = C[(C.agent == "plan_execute") & (C.model == m)].sort_values("L")
    if len(p):
        axes[0].plot(p.L, p.observed, color=col, ls=":", marker="s", ms=4, mec=SURFACE, lw=1.6, label=f"{m} P&E")
        axes[1].plot(p.L, p.implied_s_L, color=col, ls=":", marker="s", ms=4, mec=SURFACE, lw=1.6, label=f"{m} P&E")
for ax, t, yl in ((axes[0], "A. Chain success: measured vs predicted", "pass@1"),
                  (axes[1], "B. Implied per-message success", "observed^(1/L)")):
    ax.set_title(t, fontsize=9.5)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("messages in the chain (L)")
    ax.set_ylabel(yl)
    ax.set_ylim(-0.03, 1.05)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.2))
save(fig, "fig4_chain.png")

# ---------------- Fig 5: run budget (ReAct cells with 10 runs)
F = pd.read_csv("analysis/e5_ranking_flips.csv")
F = F[F.agent == "react"]
T = pd.read_csv("analysis/e5_ranking_stability.csv")
fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
for ci, c in enumerate(COND_ORDER):
    s = F[F.cond == c].sort_values("n_runs")
    if len(s):
        axes[0].plot(s.n_runs, s.reversed_or_tied_rate, color=COND_COL[c], marker=MARKERS[ci], ms=4, mec=SURFACE,
                     lw=1.6, label=c)
axes[0].set_title("A. 120b-vs-20b ordering reversed or tied", fontsize=9.5)
axes[0].set_ylabel("share of subsets")
axes[0].legend(fontsize=7, loc="upper right")
axes[1].fill_between(T.n_runs, T.tau_min, T.tau_mean, color=GRID)
axes[1].plot(T.n_runs, T.tau_mean, color="#52514e", marker="o", ms=4, mec=SURFACE, label="mean")
axes[1].plot(T.n_runs, T.tau_min, color="#e34948", marker="v", ms=4, mec=SURFACE, label="worst subset")
axes[1].axhline(0.9, color="#52514e", ls="--", lw=1)
axes[1].set_title("B. Ranking of all cells vs full-data ranking", fontsize=9.5)
axes[1].set_ylabel("Kendall tau-b")
axes[1].legend(fontsize=7.5, loc="lower right")
for ax in axes:
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("runs per task used")
axes[0].set_ylim(-0.03, 1.0)
axes[1].set_ylim(0.45, 1.03)
save(fig, "fig5_budget.png")


# ---------------- Fig 6 / 7: classifier and detector AUCs (leave-one-task-out)
def auc_fig(path, featnames, name, subsets):
    A = pd.read_csv(path)
    A = A[A.iloc[:, 0] == "leave-one-task-out"]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    combos = [("logreg", featnames[0], "#8a8985"), ("logreg", featnames[1], "#2a78d6"),
              ("gboost", featnames[0], "#c9c8c3"), ("gboost", featnames[1], "#eb6834")]
    x = np.arange(len(subsets))
    for i, (mdl, ft, col) in enumerate(combos):
        rows = A[(A.model == mdl) & (A.features == ft)].set_index("subset").reindex([s for s, _ in subsets])
        pos = x + (i - 1.5) * 0.2
        ax.bar(pos, rows.roc_auc, 0.18, color=col, label=f"{mdl}, {ft}")
        ax.errorbar(pos, rows.roc_auc, yerr=[rows.roc_auc - rows.auc_lo, rows.auc_hi - rows.roc_auc], fmt="none",
                    ecolor="#4a4a48", elinewidth=1, capsize=2)
    ax.axhline(0.5, color="#52514e", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in subsets])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, name)


auc_fig("analysis/task_classifier_scores.csv", ("context", "context+task"), "fig6_classifier.png",
        [("all", "All"), ("single", "Single tasks"), ("chain", "Chains")])
auc_fig("analysis/trace_monitor_scores.csv", ("context", "context+trace"), "fig7_detector.png",
        [("all", "All"), ("single", "Single tasks"), ("chains", "Chains")])
print("figures written to", OUT)

"""task_classifier.py (Member C): task-reliability classifier, split BY TASK.

Question: from what a task asks (known BEFORE any run) plus the run setting, can we predict whether the agent will be
UNRELIABLE on it, i.e. fail at least one of its 5 trials (pass^5 < 1)?  Logs are pooled from all conditions.

  unit      = one (agent, model, condition, fault rate, chain length, task, paraphrase) with its 5 trials
  label     = 1 if any trial failed (not pass^n), else 0
  features  = task descriptors computed from the task text, assertions and the base database
              (n_assertions, n_orders, parts that require NOT acting, wrong-customer request, social-pressure wording,
               needs a lookup/arithmetic, amount near the policy-3 limit, words, chain length, ...)
              + run setting (model size, paraphrase flag, fault rate, agent)
              Chains use the member tasks' descriptors (sums / maxima), not the wrapper text.
  split     = strictly BY TASK: when a task is held out, every training unit that contains that task is removed too
              (a held-out single task also removes chains that include it, and a held-out chain removes its members)

Evaluation: leave-one-task-out CV (the number to trust) and one seeded fixed split (the saved model); ROC-AUC, average
precision and Brier score; 95% intervals bootstrap the held-out tasks. Baseline: the same model on the run setting only.
Finally the model trained on everything scores C01-C10 (no logs exist for them yet) -> analysis/c_task_predictions.csv.

  python task_classifier.py            (run from the project root; reads results/*.jsonl, tasks/, data/)
"""
import argparse
import json
import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import chains as chains_mod
import metrics as M
from contract import get_path, load_base_db
from paraphraser import entities
from taskset import load_all_tasks

TASK_FEATS = ["n_assertions", "n_orders", "refusal_parts", "state_changes", "wrong_customer", "social_pressure",
              "needs_lookup", "near_limit", "n_amounts", "words", "n_request_types", "chain_len"]
CONTEXT_FEATS = ["is_120b", "is_paraphrase", "fault_rate", "is_plan_execute"]
FEATURE_SETS = {"context": CONTEXT_FEATS, "context+task": CONTEXT_FEATS + TASK_FEATS}
SOCIAL = re.compile(r"\b(friend|neighbou?r|behalf|asked me|wife|husband|sister|brother|colleague|cousin)\b", re.I)
LOOKUP = re.compile(r"\b(half|full amount|the full|whole|everything|all of it)\b", re.I)


def task_features(task: dict, base: dict, text: str | None = None) -> dict:
    """Descriptors of ONE single task (no logs involved). `text` overrides the instruction (paraphrases)."""
    text = text or task["instruction"]
    low = text.lower()
    assertions = task["assertions"]
    orders = {a["path"].split(".")[1] for a in assertions}
    speaker = re.search(r"\bC\d\b", text)
    speaker = speaker.group(0) if speaker else None
    owners = {base["orders"][o]["customer_id"] for o in orders if o in base["orders"]}
    refusal_parts = sum(1 for a in assertions if get_path(base, a["path"]) == a["equals"])
    nums = [float(x) for x in entities(text)["numbers"]]
    return {
        "n_assertions": len(assertions),
        "n_orders": len(orders),
        "refusal_parts": refusal_parts,                      # assertions the base state already satisfies = "do not act"
        "state_changes": len(assertions) - refusal_parts,    # assertions that need a real tool call
        "wrong_customer": int(speaker is not None and any(o != speaker for o in owners)),
        "social_pressure": int(bool(SOCIAL.search(text))),
        "needs_lookup": int(bool(LOOKUP.search(text))),
        "near_limit": int(any(90 <= n <= 110 for n in nums)),  # around the policy-3 limit of 100
        "n_amounts": len(nums),
        "words": len(text.split()),
        "n_request_types": sum(w in low for w in ("cancel", "refund", "address")),
        "chain_len": 1,
    }


def chain_features(chain: dict, tasks_by_id: dict, base: dict) -> dict:
    parts = [task_features(tasks_by_id[m], base) for m in chain["chain"]]
    out = {}
    for k in TASK_FEATS:
        vals = [p[k] for p in parts]
        out[k] = max(vals) if k in ("wrong_customer", "social_pressure", "needs_lookup", "near_limit", "n_request_types") else sum(vals)
    out["chain_len"] = len(chain["chain"])
    return out


def build_units(df: pd.DataFrame, tasks: list, chains: list, base: dict) -> pd.DataFrame:
    tasks_by_id = {t["id"]: t for t in tasks}
    chains_by_id = {c["id"]: c for c in chains}
    u = M.unit_table(df)
    rows = []
    for _, r in u.iterrows():
        tid = r["task_id"]
        if r["condition"] == "chain":
            if tid not in chains_by_id:
                continue
            feats = chain_features(chains_by_id[tid], tasks_by_id, base)
            members = set(chains_by_id[tid]["chain"])
        else:
            if tid not in tasks_by_id:
                continue
            t = tasks_by_id[tid]
            text = None
            if r["condition"] == "paraphrase" and r["paraphrase_idx"] == r["paraphrase_idx"]:
                text = t["paraphrases"][int(r["paraphrase_idx"])]
            feats = task_features(t, base, text)
            members = {tid}
        rows.append({"agent": r["agent"], "model": r["model"], "condition": r["condition"], "fault_rate": r["fault_rate"],
                     "task_id": tid, "paraphrase_idx": r["paraphrase_idx"], "n": int(r["n"]), "c": int(r["c"]),
                     "unreliable": int(r["c"] < r["n"]), "pass1": r["c"] / r["n"], "members": ",".join(sorted(members)),
                     "is_120b": int(r["model"].endswith("120b")), "is_paraphrase": int(r["condition"] == "paraphrase"),
                     "is_plan_execute": int(r["agent"] != "react"), **feats})
    return pd.DataFrame(rows)


def make_model(name: str):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000))
    return GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.1, subsample=0.8, random_state=0)


def member_sets(units):
    return [set(s.split(",")) for s in units["members"]]


def fit_predict(train, test, feats, model_name):
    m = make_model(model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(train[feats].to_numpy(float), train["unreliable"].to_numpy())
    return m, m.predict_proba(test[feats].to_numpy(float))[:, 1]


def loto(units, feats, model_name):
    """Out-of-fold P(unreliable). Group = task (single) or chain id; training excludes every unit that shares a task."""
    p = np.zeros(len(units))
    sets = member_sets(units)
    for g in units["task_id"].unique():
        te = (units["task_id"] == g).to_numpy()
        held = set().union(*[s for s, t in zip(sets, te) if t])
        tr = np.array([not (s & held) for s in sets])
        _, p[te] = fit_predict(units[tr], units[te], feats, model_name)
    return p


def block(y, p) -> dict:
    out = {"units": int(len(y)), "unreliable": int(np.sum(y))}
    if len(set(y)) < 2:
        return {**out, "roc_auc": np.nan, "avg_precision": np.nan, "brier": brier_score_loss(y, p)}
    return {**out, "roc_auc": roc_auc_score(y, p), "avg_precision": average_precision_score(y, p),
            "brier": brier_score_loss(y, p)}


def boot_auc(d, pcol, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    ys = [g["unreliable"].to_numpy() for _, g in d.groupby("task_id")]
    ps = [g[pcol].to_numpy() for _, g in d.groupby("task_id")]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ys), size=len(ys))
        y = np.concatenate([ys[i] for i in pick])
        if y.min() == y.max():
            continue
        vals.append(roc_auc_score(y, np.concatenate([ps[i] for i in pick])))
    return (np.nan, np.nan) if len(vals) < 50 else tuple(np.percentile(vals, [2.5, 97.5]))


def fixed_split(units, test_frac=0.25, min_pos=4, max_tries=2000):
    groups = sorted(units["task_id"].unique())
    sets = member_sets(units)
    for seed in range(max_tries):
        rng = np.random.default_rng(seed)
        test_g = {str(g) for g in rng.choice(groups, size=max(3, round(len(groups) * test_frac)), replace=False)}
        te = units["task_id"].isin(test_g).to_numpy()
        held = set().union(*[s for s, t in zip(sets, te) if t])
        tr = np.array([not (s & held) for s in sets])
        yt, ytr = units.loc[te, "unreliable"], units.loc[tr, "unreliable"]
        if yt.sum() >= min_pos and (1 - yt).sum() >= min_pos and ytr.sum() >= 2 * min_pos and (1 - ytr).sum() >= 2 * min_pos \
                and units.loc[te, "condition"].eq("chain").any() and (~units.loc[te, "condition"].eq("chain")).sum() >= 8:
            return seed, sorted(test_g), tr, te
    raise RuntimeError("no acceptable split")


def fig_task_classifier(t, figdir, note=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    plt.rcParams.update({"figure.facecolor": surf, "axes.facecolor": surf, "savefig.facecolor": surf, "axes.edgecolor": grid,
                         "axes.labelcolor": ink2, "text.color": ink, "xtick.color": ink2, "ytick.color": ink2,
                         "axes.grid": True, "grid.color": grid, "axes.axisbelow": True, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "axes.titleweight": "bold",
                         "axes.titlelocation": "left"})
    cv = t[t["eval"] == "leave-one-task-out"]
    subsets = ["all", "single", "chain"]
    names = {"all": "All units", "single": "Single-task units", "chain": "Chain units"}
    combos = [("logreg", "context", "#8a8985"), ("logreg", "context+task", "#2a78d6"),
              ("gboost", "context", "#c9c8c3"), ("gboost", "context+task", "#eb6834")]
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
    ax.set_xticks(range(3))
    ax.set_xticklabels([names[s] for s in subsets])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("ROC-AUC for 'unreliable' (pass^5 < 1)")
    ax.set_title("Task-reliability classifier, leave-one-task-out: task descriptors add signal over run setting alone", loc="left", fontsize=10)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.01, 0.005, "Dashed line = chance. 95% intervals bootstrap held-out tasks. " + note, fontsize=7.5, color=ink2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, "fig5_task_classifier_auc.png"), dpi=170)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", default=None)
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    logs = a.logs or M.default_logs()
    df = M.load_logs(logs)
    base = load_base_db()
    tasks = load_all_tasks()
    chains = chains_mod.load_chains()
    units = build_units(df, tasks, chains, base)
    print(f"{len(logs)} log files, {len(df)} episodes -> {len(units)} units from {units['task_id'].nunique()} tasks/chains; "
          f"{units['unreliable'].sum()} unreliable ({units['unreliable'].mean():.0%})")
    print(units.groupby("condition")["unreliable"].agg(["size", "sum"]).rename(columns={"size": "units", "sum": "unreliable"}))

    rows = []
    for mn in ("logreg", "gboost"):
        for fs, feats in FEATURE_SETS.items():
            units[f"p_{mn}_{fs}"] = loto(units, feats, mn)
            for subset in ("all", "single", "chain"):
                s = units if subset == "all" else units[(units["condition"] == "chain") == (subset == "chain")]
                lo, hi = boot_auc(s, f"p_{mn}_{fs}")
                rows.append({"eval": "leave-one-task-out", "model": mn, "features": fs, "subset": subset,
                             **block(s["unreliable"].to_numpy(), s[f"p_{mn}_{fs}"].to_numpy()), "auc_lo": lo, "auc_hi": hi})
    seed, test_groups, tr, te = fixed_split(units)
    print(f"\nfixed split by task (seed {seed}): test tasks/chains = {test_groups}; "
          f"train units {int(tr.sum())}, test units {int(te.sum())}")
    models = {}
    for mn in ("logreg", "gboost"):
        for fs, feats in FEATURE_SETS.items():
            m, p = fit_predict(units[tr], units[te], feats, mn)
            models[(mn, fs)] = m
            d = units[te].copy()
            d["p"] = p
            for subset in ("all", "single", "chain"):
                s = d if subset == "all" else d[(d["condition"] == "chain") == (subset == "chain")]
                lo, hi = boot_auc(s.rename(columns={"p": "pp"}), "pp") if s["task_id"].nunique() >= 4 else (np.nan, np.nan)
                rows.append({"eval": f"fixed-split(seed={seed})", "model": mn, "features": fs, "subset": subset,
                             **block(s["unreliable"].to_numpy(), s["p"].to_numpy()), "auc_lo": lo, "auc_hi": hi})
    t = pd.DataFrame(rows)
    num = t.select_dtypes("number").columns
    t[num] = t[num].round(3)
    t.to_csv(os.path.join(a.out, "task_classifier_scores.csv"), index=False)
    fig_task_classifier(t, os.path.join(a.out, "figures"),
                        f"{len(units)} units, {int(units['unreliable'].sum())} unreliable, {units['task_id'].nunique()} tasks/chains.")
    units.drop(columns=[c for c in units.columns if c.startswith("p_")] + ["members"]).assign(
        **{c: units[c] for c in units.columns if c.startswith("p_")}).to_csv(
        os.path.join(a.out, "task_classifier_units_loto.csv"), index=False)

    # saved models: fixed-split logistic (train tasks only) + final models trained on every unit
    feats = FEATURE_SETS["context+task"]
    joblib.dump({"model": models[("logreg", "context+task")], "features": feats, "test_tasks": test_groups, "split_seed": seed,
                 "note": "P(agent fails at least one of 5 trials) from task descriptors + run setting; trained on train tasks only"},
                os.path.join(a.out, "task_reliability_clf.joblib"))
    final = {mn: fit_predict(units, units.head(1), feats, mn)[0] for mn in ("logreg", "gboost")}
    joblib.dump({"models": final, "features": feats}, os.path.join(a.out, "task_reliability_final.joblib"))
    coefs = dict(zip(feats, final["logreg"][-1].coef_[0].round(3).tolist()))
    json.dump({"split_seed": seed, "test_tasks": test_groups, "logreg_standardized_coefficients(+ = more likely unreliable)": coefs},
              open(os.path.join(a.out, "task_classifier_coefs.json"), "w"), indent=1)

    # predictions for tasks that have no logs yet (C01-C10), clean condition, react agent
    have = set(units["task_id"])
    pred_rows = []
    for tk in tasks:
        if tk["id"] in have or tk["owner"] != "C":
            continue
        f = task_features(tk, base)
        for model_name, is120 in (("gpt-oss-120b", 1), ("gpt-oss-20b", 0)):
            x = {**f, "is_120b": is120, "is_paraphrase": 0, "fault_rate": 0.0, "is_plan_execute": 0}
            X = np.array([[x[k] for k in feats]], dtype=float)
            pred_rows.append({"task_id": tk["id"], "model": model_name, "refusal": bool(tk.get("refusal")),
                              "p_unreliable_logreg": round(float(final["logreg"].predict_proba(X)[0, 1]), 3),
                              "p_unreliable_gboost": round(float(final["gboost"].predict_proba(X)[0, 1]), 3),
                              **{k: f[k] for k in ("n_assertions", "wrong_customer", "social_pressure", "needs_lookup", "near_limit")}})
    pd.DataFrame(pred_rows).to_csv(os.path.join(a.out, "c_task_predictions.csv"), index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n", t.to_string(index=False))
    print("\nlogreg coefficients:", coefs)
    print("\nC-task predictions (not validated: no C logs exist):\n", pd.DataFrame(pred_rows).to_string(index=False))
    return t, units


if __name__ == "__main__":
    main()

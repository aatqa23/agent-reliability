# Member C: what is in this package

| Your list item | File(s) | Status |
|---|---|---|
| 1. Tasks C01-C10 | `tasks/C.json`, `solve_tasks_c.py` (hand-solve check), `data/paraphrases.json` (C01-C10 added) | done, all checks pass |
| 2. Second agent | `agents/plan_execute.py`, registered in `agents/__init__.py` as `AGENTS["plan_execute"]`, `test_plan_execute.py` | done, **tested offline only** (no live runs: no Groq key here) |
| 3. Metrics, E5, classifier, chain decay, plots | `metrics.py`, `analyze_reliability.py`, `task_classifier.py`, `trace_monitor.py`, `test_metrics.py`, `test_task_classifier.py` | done, run on all logs in `results/*.jsonl` (680 episodes) |

Outputs are in `analysis/` (CSVs, saved models, `figures/fig1-6`).

## Setup (Windows: see the next section)
1. Copy this folder over the merged project folder (A's files + B's files). Do not copy anyone's `.env` or `.venv` (A's `.venv` was made on a Mac and does not work on Windows: delete it).
2. Activate the venv, then: `pip install numpy pandas scipy scikit-learn matplotlib joblib openai pydantic tqdm python-dotenv`
3. Offline checks (no API calls): `python check_all.py` (runs all 14 checks; same on Windows, Mac and Linux)
Note: B's `test_b.py` asserts exactly 20 tasks, so it fails once `tasks/C.json` exists. That is its hard-coded count, not a bug.

## Analysis (from the project root; each pools every `results/*.jsonl`, newest line wins, api_error/crash rows dropped)
```
python analyze_reliability.py   # pass@k, pass^k + CIs, E5, chain decay, fig1-4   (~10 s)
python task_classifier.py       # task-reliability classifier split by task, fig5   (~20 s)
python trace_monitor.py         # trace-based failure monitor, fig6                  (~30 s)
```
With two agents in the logs every table gets an agent column (agents are never pooled), figures are drawn per agent, and
`analysis/agent_comparison.csv` holds paired differences (react minus plan_execute, bootstrap over tasks).

## Live runs still to do (each needs your own Groq key; same settings: temperature 0.7, 5 seeds, `--workers 1 --delay 1`)
```
python run_e1.py --tasks tasks/C.json --out results/e1_c.jsonl --workers 1 --delay 1                      # react on C01-C10 (100 episodes)
python run_e1.py --tasks tasks/B.json --agents plan_execute --out results/e1_pe_b.jsonl --workers 1 --delay 1   # second agent, clean, B tasks
python run_experiments.py e4 --agents plan_execute --lens 4 --per-len 4 --workers 1 --out results/e4_pe_len4.jsonl   # does the audit fix long chains?
```
The third command is the most informative (react collapses at length 4). plan_execute uses 2 extra LLM calls per episode, so mind the 200k tokens/day/model limit. After each run, re-run the three analysis commands.

## What was found (react agent, 680 episodes; clean = 20 A+B tasks, other conditions B tasks, chains 4 per length)
- **Clean:** 120b pass@1 1.00, 20b 0.99 (pass^5 1.00 / 0.95). Faults hurt pass^5 much more than pass@1 (20b at rate 0.2: 0.82 vs 0.50).
- **E5 (only 5 seeds exist, so "2 to 10 trials" is covered up to 5):** the ranking of all 12 (model, condition) cells matches the full ranking from 4 runs (worst subset Kendall tau >= 0.9); with 1 run the mean tau is 0.74. The 120b-vs-20b ordering is reversed or tied in 40% of single-run draws for paraphrase and fault 0.1, and 80% for 4-message chains.
- **Chain decay:** independence with p = clean pass@1 predicts 1.00 (120b) and 0.96 (20b) at 4 messages. Observed: 0.05 and 0.00. Implied per-message success: 0.95 -> 0.47 (120b), 0.71 -> 0.00 (20b). Only 4 chains per length, so the direction is clear and the size is not.
- **Task-reliability classifier** (unit = task x model x condition, label = pass^5 < 1; 136 units, 29 unreliable; leave-one-task-out): task descriptors lift ROC-AUC from 0.55 (run setting only) to 0.82 [0.70, 0.95] with gradient boosting, 0.74 on single tasks only; logistic regression is weaker (0.70). The fixed split by task (7 test tasks, 7 unreliable units) gives 0.83 but its interval is [0.42, 1.0]: trust leave-one-task-out.
- **Trace monitor** (episode-level): on chains, trace features lift AUC from 0.87-0.88 to 0.96-0.98 (a short trace = early stop); on single tasks it is near chance (0.53-0.65, 27 failures).
- `analysis/c_task_predictions.csv` scores C01-C10 before any run. The two models disagree strongly (C10: 0.70 vs 0.04), so treat it as a prior to compare with real C results, not as evidence.

## Limits to state in the report
- Small samples: 10 B tasks for paraphrase/faults, 4 chains per length, 5 seeds. Intervals are bootstrap over tasks, or Wilson with n = number of tasks/chains when there are fewer than 8 or all results are equal.
- Chains mix A and B tasks; chain-vs-single comparisons are not like for like.
- The classifier's 28 tasks/chains give wide intervals, and the split removes every training unit that shares a task with the test unit (chains included).
- C01-C10 have not been run, and the second agent has never called a real model.

# Agent reliability project: final submission

Group project (Members A, B, C): how reliable are LLM agents? Two models (gpt-oss-120b, gpt-oss-20b), two agent
designs (ReAct, plan-then-execute), 30 tasks, faults, paraphrases, request chains, pass@k / pass^k.

**Start with `Report.docx`** (same content as `Report.pdf`). It has the results, figures, limits and appendices.

## What is where

| Path | Contents |
|---|---|
| `Report.docx`, `Report.pdf` | the report |
| `results/*.jsonl` | raw per-episode logs: `e1.jsonl` (A tasks), `e1_b.jsonl` (B), `e1_c.jsonl` (C), `e2.jsonl` paraphrases, `e3.jsonl` faults, `e4.jsonl` chains, `e1_pe_b.jsonl` and `e4_pe_len4.jsonl` (plan-then-execute) |
| `analysis/` | all tables (`*.csv`) and `figures/fig1` to `fig6`, regenerated from `results/` |
| `tasks/` | `A.json`, `B.json`, `C.json` (30 tasks) |
| `data/` | frozen paraphrases (`paraphrases.json`) and chains (`chains.json`) |
| `env/`, `contract.py`, `CONTRACT.md` | simulated shop, tools, policies, task validator |
| `agents/` | `react.py` (A), `plan_execute.py` (C) |
| `llm.py`, `run_episode.py`, `run_e1.py`, `run_experiments.py` | model client with disk cache, episode runner, experiment runners |
| `faults.py`, `chains.py`, `paraphraser.py`, `taskset.py` | fault injector, chain builder, paraphraser, task loader (B) |
| `metrics.py`, `analyze_reliability.py`, `task_classifier.py`, `trace_monitor.py` | metrics, E5 and chain-decay analysis, plots, classifiers (C) |
| `docs/` | each member's own notes (`README_B.md`, `README_C.md`) and the reduced run plan |

## Setup (only needed to re-run things)

Python 3.10+.

Windows (PowerShell):

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Mac / Linux:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

A Groq key is only needed to run new episodes: create `.env` with `GROQ_API_KEY=your_key`.
The submission does not include a key.

## Re-running

No API key needed:

    python check_all.py                 # 14 offline checks, all should print [ok]
    python analyze_reliability.py       # tables and fig1 to fig4 in analysis/
    python task_classifier.py           # fig5 and classifier scores
    python trace_monitor.py             # fig6 and monitor scores

These read every `results/*.jsonl` file, so the numbers in the report are reproduced exactly.

New live runs (need the key; free-tier Groq allows 200,000 tokens per model per day, so use `--workers 1 --delay 1`):

    python run_e1.py --tasks tasks/C.json --out results/e1_c.jsonl --workers 1 --delay 1
    python run_e1.py --tasks tasks/B.json --agents plan_execute --out results/e1_pe_b.jsonl --workers 1 --delay 1
    python run_experiments.py e4 --agents plan_execute --lens 4 --per-len 4 --workers 1 --out results/e4_pe_len4.jsonl

A run that stops can be started again with the same command; finished episodes are skipped.

## Known limits

- `plan_execute` on 4-request chains is incomplete (11 usable episodes of 40 planned) because of the daily token limit.
- `test_b.py` is B's original test and assumes exactly 20 tasks, so it fails now that the C tasks exist. It is not part of `check_all.py`.
- Chains were built from the A and B tasks only.
- The model-reply cache (`results/cache/`) is not included because of its size; reruns of the original episodes would call the API again.

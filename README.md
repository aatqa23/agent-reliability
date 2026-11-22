# Agent reliability project: final submission

Group project (Members A, B, C): how reliable are LLM agents? Two models (gpt-oss-120b, gpt-oss-20b), plus a third
from another family (qwen3.8-27b) since phase 3, two agent designs (ReAct, plan-then-execute), 30 tasks, faults,
paraphrases, request chains, pass@k / pass^k.

**Start with `Report.docx`** (same content as `Report.pdf`). It has the results, figures, limits and appendices.

## What is where

| Path | Contents |
|---|---|
| `Report.docx`, `Report.pdf` | the report |
| `results/sample/` | small cached sample (18 episodes + every model reply they needed) for offline reruns, see below |
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

    python check_all.py                 # 16 offline checks, all should print [ok]
    python replay_sample.py --check     # (also part of check_all) re-runs the sample episodes from the cache
    python analyze_reliability.py       # tables and fig1 to fig4 in analysis/
    python task_classifier.py           # fig5 and classifier scores
    python trace_monitor.py             # fig6 and monitor scores

These read every `results/*.jsonl` file, so the numbers in the report are reproduced exactly.

New live runs (need the key; free-tier Groq allows 200,000 tokens per model per day, so use `--workers 1 --delay 1`,
use your own key, and spread big runs over several days; a stopped run resumes where it left off):

    python run_e1.py --tasks tasks/C.json --out results/e1_c.jsonl --workers 1 --delay 1
    python run_e1.py --tasks tasks/B.json --agents plan_execute --out results/e1_pe_b.jsonl --workers 1 --delay 1
    python run_experiments.py e4 --agents plan_execute --lens 4 --per-len 4 --workers 1 --out results/e4_pe_len4.jsonl

A run that stops can be started again with the same command; finished episodes are skipped.

`--models` takes full model ids or the aliases in `llm.MODELS`: `large` (gpt-oss-120b), `small` (gpt-oss-20b),
`qwen` (qwen/qwen3.8-27b). The default is still the two gpt-oss models. Example for the third model:

    python run_e1.py --tasks tasks/A.json --models qwen --agents react plan_execute --out results/e1_qwen_a.jsonl --workers 1 --delay 1

Budget guide from the sample (mean tokens per single-task episode): ReAct about 2,200 (gpt-oss) and 4,400 (qwen);
plan-then-execute about 5,100 (gpt-oss) and 8,400 (qwen). So one day of a 200k key is roughly 80 gpt-oss or 45 qwen
ReAct episodes, or 35 gpt-oss or 20 qwen plan-then-execute episodes. Use `run_experiments.py ... --dry-run` to count
episodes before starting.

### Offline reruns from a cache

`llm.py` caches every reply in `results/cache/` (key = messages, model, temperature, seed, tools, reasoning effort).
With `LLM_OFFLINE=1` the client never calls the API: a reply missing from the cache raises `CacheMiss` at once.
`LLM_CACHE_DIR` points it at another cache. The full cache is not in the repo (size), so `results/sample/` ships a
small one: 3 tasks (A01, A04, C03) x 2 agents x 3 models, seed 0.

    python replay_sample.py --check          # offline, identical success/steps/tokens for all 18 episodes
    python replay_sample.py --record         # live: rebuild the sample (needs a key)

To make your own run replayable offline, keep its cache directory next to the results:

    LLM_CACHE_DIR=results/my_cache python run_e1.py ... --out results/my_run.jsonl           # live, fills the cache
    LLM_CACHE_DIR=results/my_cache LLM_OFFLINE=1 python run_e1.py ... --out /tmp/replay.jsonl   # offline replay

(Windows PowerShell: `$env:LLM_OFFLINE="1"` before the command.)

## Episode log fields

Every line in `results/*.jsonl` is one `EpisodeLog` (see `contract.py`). The original fields are unchanged:
`task_id, agent, model, condition, paraphrase_idx, fault_rate, chain_len, seed, success, steps, tool_calls, error,
latency_s`. Phase 3 only appends new fields; rows written before phase 3 do not have them (read them as missing):

| Field | Meaning |
|---|---|
| `wall_clock_s` | wall-clock seconds for the whole episode (setup, agent loop, assertion check); `latency_s` is the agent loop only |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | summed over all LLM calls of the episode, as reported by the API. Cached calls report the usage stored when they ran live, so this is the cost of the episode |
| `live_tokens` | tokens actually spent in this run (0 when every reply came from the cache) |
| `api_calls`, `cache_hits` | live calls and cache hits in the episode |
| `untracked_calls` | cache hits from entries written before token accounting existed (their tokens are unknown, counted as 0) |
| `plan_ok` | plan-then-execute only: a valid plan reached the executor (`null` for ReAct) |

Cost per run = sum of `total_tokens` over its rows. Tokens Groq spends on generations it rejects
(400 `tool_use_failed`) are not returned by the API and are not counted, so episodes with such retries under-report
slightly; an episode whose every call was rejected shows 0 tokens.

## Phase 3 changes (A)

- **Plan-then-execute planning call fixed.** The planner reused the executor's system prompt ("call exactly ONE
  function") but was sent without tools, so gpt-oss tried to call a tool and Groq answered every attempt with
  400 "Tool choice is none, but model called a tool". After 3 retries the client handed back the raw generation, and
  that (a tool-call JSON or a premature final reply) was inserted as "the plan". The planner now has its own prompt
  (`build_plan_prompt()`: policies + tool list as reference, no function-calling format); a reply that is empty or
  looks like a function call is rejected and asked for once more; if both fail the episode runs without a plan and
  logs `plan_ok=false`. Live check: 9/9 plans valid (3 tasks x 3 models), no 400s.
  **Consequence:** `results/e1_pe_b.jsonl` and `results/e4_pe_len4.jsonl` were produced with the broken planner.
  Re-run plan-then-execute into NEW files (for example `results/e1_pe_b_v2.jsonl`), because the runner skips episodes
  that already exist in the output file, and move the old files out of `results/` (e.g. to `results/pre_fix/`) when
  the new ones are complete so the analysis does not pool both.
- **Planner no longer guesses order facts (second fix).** With only the fix above, plans were valid prose but stated
  outcomes the planner could not know yet (e.g. "O1002 belongs to Bilal -> yes", "total is $150") and pre-wrote the
  final reply; the executor followed those guesses, skipped get_order and refused valid requests. The planning prompt
  now says no order data has been seen, and asks for each request: get_order, then "ACT only if <owner, status,
  limits>", otherwise REFUSE; a pre-written final reply is cut off (`clean_plan()`). Seed 0 on the four length-4
  chains: gpt-oss-20b 0/2 (first fix only) -> 3/4, gpt-oss-120b 3/4. Caveat: this was found by reading traces of the
  same chains that are evaluated, so treat it as a design iteration, and use only runs made with this version
  (commit after 5100997) for plan-then-execute results.
- **Third model, other family:** `qwen/qwen3.8-27b` (alias `qwen`), same client, temperature 0.7, tools, policies and
  `reasoning_effort="low"`. It was the only non-gpt-oss chat model available on Groq for our key (the Llama models
  Groq lists are prompt-guard classifiers, not chat models).
- **Tokens and wall-clock time per episode** in the log (fields above). Existing field names are unchanged.
- **Reproducibility:** `LLM_OFFLINE=1`, the cached sample in `results/sample/`, `replay_sample.py`, and
  `test_phase3.py`; both are in `check_all.py`. The `tool_use_failed` fallback text is now also written to the cache,
  flagged `"fallback": true`. Live runs still ignore it and retry (as before); only offline replays use it, so a
  replay reproduces such episodes exactly.

## Known limits

- `plan_execute` on 4-request chains is incomplete (11 usable episodes of 40 planned) because of the daily token limit.
- `test_b.py` is B's original test and assumes exactly 20 tasks, so it fails now that the C tasks exist. It is not part of `check_all.py`.
- Chains were built from the A and B tasks only.
- The full model-reply cache (`results/cache/`) is not included because of its size; reruns of the original episodes
  would call the API again. `results/sample/` is a small cache that reruns offline (see above).
- Rows written before phase 3 have no token or wall-clock fields; cache entries written before phase 3 have no usage.

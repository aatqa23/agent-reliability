# Member B: what is in this package

| Your list item | File(s) |
|---|---|
| 1. Tasks B01-B10 (4 refusals + 1 mixed, 3 multi-step, 11 of 12 orders used) | `tasks/B.json`, hand-solve check `solve_tasks_b.py` |
| 2. Disk cache | `llm.py` (same `chat(...)` signature; cache in `results/cache/`) |
| 3. Fault injector | `faults.py` (`make_fault_hook(rate, rng)`) |
| 4. Paraphraser (3 per task) | `paraphraser.py`, `taskset.py`, `data/paraphrases.json` (A01-A10, B01-B10 done) |
| 5. Chain composer (length 2-4) | `chains.py`, `data/chains.json` (30 chains: 10 each of length 2, 3, 4) |
| 6. E2 / E3 / E4 runner | `run_experiments.py`, `run_b.sh` |
| Tests (all offline) | `test_faults.py`, `test_cache.py`, `test_b.py`, `test_integration.py` |

## Setup (same as A's, then B's checks)
1. Copy this folder's contents over your project folder (same layout as A's zip). Do not copy A's `.env` or `.venv`.
2. `python -m venv .venv`, activate it, `pip install openai pydantic tqdm pandas python-dotenv`.
3. Put YOUR OWN key in `.env`: `GROQ_API_KEY=...`
4. Check everything (all offline, no API calls):
   `python contract.py && python test_tools.py && python test_react.py && python test_llm.py && python test_run.py && python solve_tasks_b.py && python test_faults.py && python test_cache.py && python test_b.py && python test_integration.py && python paraphraser.py --check && python chains.py --check`

## Running (from the project root)
- Everything, in priority order: `./run_b.sh` (or run the four commands inside it yourself).
- Preview size/time first: add `--dry-run`. Smoke test: `python run_experiments.py e3 --limit 2 --n 1 --models openai/gpt-oss-20b`.
- Results: `results/e1_b.jsonl` (clean baseline for B01-B10, needed so E2-E4 have something to compare to), `e2.jsonl`, `e3.jsonl`, `e4.jsonl`.
- Summary table any time: `python run_experiments.py summary results/e3.jsonl`.
- Size at A's settings (2 models, 5 seeds): clean 100, E2 600, E3 400, E4 300 episodes (long ones) on the 20 tasks now available. At ~12 s per single-task episode plus the 1 s delay that is roughly 7 hours, and about 9 hours once C's tasks are added. Free-tier limits may make it slower, so start tonight.

## When C's tasks (tasks/C.json) arrive
1. `python paraphraser.py` (uses your key; fills only C01-C10), read them with `python paraphraser.py --show`, then `python paraphraser.py --check`.
2. If E4 has not started yet: `python chains.py --build` so chains also use C's tasks. If E4 already ran, keep it (valid as is).
3. Re-run the same E2/E3 commands: finished episodes are skipped, only C's tasks are added.

## Things to tell A (small deviations from the brief)
- **Paraphrases are in `data/paraphrases.json`, not `tasks/`.** `python contract.py` reads every `tasks/*.json` as a list of tasks and would crash on the dictionary. `taskset.load_all_tasks()` injects them into `task["paraphrases"]`.
- **`run_episode.py` (one change):** the agent's step budget is 15 x chain length (15 for single tasks, so E1 is unchanged). With a flat 15, long chains would fail on the cap alone.
- **`test_llm.py`:** one line added so those tests bypass the cache (`LLM_CACHE=0`).
- **Chains:** the instruction is a one-sentence wrapper ("N separate customer messages...") plus each member instruction verbatim; assertions are concatenated. Members never share an order, and a wrong-customer request is never chained with a message from that order's owner.
- **Faults:** each tool call fails with probability = rate (timeout, HTTP 429 or garbled output, equally likely). The tool is not executed, so the database is never changed by a fault.
- **Cache:** key = messages + model + temperature + seed + tools. The seed is not sent to the API (as before); it only keeps the 5 trials independent. Failures are never cached.

## Send to A by Sunday noon
`results/e1_b.jsonl`, `e2.jsonl`, `e3.jsonl`, `e4.jsonl`, plus the code (`faults.py`, `chains.py`, `paraphraser.py`, `taskset.py`, `run_experiments.py`, `llm.py`, `run_episode.py`, `data/`, `tasks/B.json`). Do not send `.env`.

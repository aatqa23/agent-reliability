# Reduced plan (Groq free tier: 200,000 tokens per model per day per account)

Same settings as before (both models, temperature 0.7, 5 seeds). Only the NUMBER OF TASKS/CHAINS is cut.
Everyone uses this same project folder (same data/chains.json, same paraphrases) and their OWN Groq key in .env.

| Who | Command (run from the project root, venv active) | Episodes | Output file |
|---|---|---|---|
| A | `python run_experiments.py e2 --tasks tasks/B.json --paraphrases 2 --workers 1 --out results/e2.jsonl` | 200 | results/e2.jsonl |
| C | `python run_experiments.py e4 --lens 4 --per-len 4 --workers 1 --out results/e4_len4.jsonl` | 40 (long) | results/e4_len4.jsonl |
| B | `python run_experiments.py e4 --lens 2 --per-len 4 --workers 1 --out results/e4_len2.jsonl` | 40 | results/e4_len2.jsonl |
| B | `python run_experiments.py e3 --tasks tasks/B.json --models openai/gpt-oss-120b --workers 1 --out results/e3_120b.jsonl` | 100 | results/e3_120b.jsonl |
| B | `python run_experiments.py e3 --tasks tasks/B.json --models openai/gpt-oss-20b --workers 1 --out results/e3_20b.jsonl` | 100 | results/e3_20b.jsonl |

If a run stops (daily limit, Ctrl-C, crash): wait / fix, then run the SAME command again. It resumes.
If you hit "tokens per day (TPD)" 429 errors: stop (Ctrl-C). That key is used up until the daily reset.

After everyone is done, B merges (merge keeps only complete 5-seed units and never overwrites):
    python run_experiments.py merge results/e3_120b.jsonl results/e3_20b.jsonl --out results/e3.jsonl --only-complete 5
    python run_experiments.py merge results/e4_len2.jsonl results/e4_len4.jsonl --out results/e4.jsonl --only-complete 5
    python run_experiments.py summary results/e1_b.jsonl results/e2.jsonl results/e3.jsonl results/e4.jsonl

Do NOT run `python chains.py --build` or `python paraphraser.py` (they would change the frozen data).

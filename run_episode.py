"""run_episode(task, agent, model, condition, seed) -> dict   (an EpisodeLog as a dict)

This is the function every experiment calls (E1-E5). See contract.py for the schema.

condition keys (all optional, defaults shown):
  name: "clean" | "paraphrase" | "fault" | "chain"      default "clean"
  paraphrase_idx: int | None                            default None  (used when name == "paraphrase")
  fault_rate: float                                     default 0.0   (> 0 needs B's faults.py)
  chain_len: int                                        default 1     (B's chain composer builds the task)

B's contract for faults.py:   make_fault_hook(rate: float, rng: random.Random) -> FaultHook
"""
import copy
import random
import time
from dataclasses import asdict

from agents import AGENTS
from contract import EpisodeLog, check_assertions, load_base_db
from env.tools import TOOLS

BASE_MAX_STEPS = 15      # agent step budget for a single task (chain of length L gets L x this; see below)
_BASE = None


def _base() -> dict:
    global _BASE
    if _BASE is None:
        _BASE = load_base_db()
    return _BASE


def default_condition() -> dict:
    return {"name": "clean", "paraphrase_idx": None, "fault_rate": 0.0, "chain_len": 1}


def run_episode(task: dict, agent: str, model: str, condition: dict | None = None, seed: int = 0) -> dict:
    import llm
    t_start = time.time()
    llm.usage_reset()              # per-thread token counter for this episode
    cond = {**default_condition(), **(condition or {})}
    db = copy.deepcopy(_base())

    instruction = task["instruction"]
    if cond["name"] == "paraphrase":
        idx = cond["paraphrase_idx"]
        if idx is None or idx >= len(task.get("paraphrases", [])):
            raise ValueError(f"task {task['id']} has no paraphrase {idx}")
        instruction = task["paraphrases"][idx]

    hook = None
    if cond["fault_rate"] > 0:
        from faults import make_fault_hook          # B's module
        rng = random.Random(f"{task['id']}|{seed}|{cond['fault_rate']}")   # reproducible per episode
        hook = make_fault_hook(cond["fault_rate"], rng)

    t0 = time.time()
    # B: the step budget grows with chain length, otherwise a long chain would fail on the step cap alone
    # (a 4-task chain needs 8-15 steps even for a perfect agent). chain_len = 1 -> 15, exactly as in E1.
    max_steps = BASE_MAX_STEPS * max(1, int(cond["chain_len"]))
    res = AGENTS[agent](instruction, db, TOOLS, model, seed, fault_hook=hook, max_steps=max_steps)
    latency = time.time() - t0

    log = EpisodeLog(
        task_id=task["id"], agent=agent, model=model, condition=cond["name"],
        paraphrase_idx=cond["paraphrase_idx"], fault_rate=float(cond["fault_rate"]),
        chain_len=int(cond["chain_len"]), seed=seed,
        success=check_assertions(db, task["assertions"]),
        steps=res["steps"], tool_calls=res["tool_calls"], error=res["error"],
        latency_s=round(latency, 3),
        plan_ok=res.get("plan_ok"),
        **llm.usage_get(),
    )
    log.wall_clock_s = round(time.time() - t_start, 3)
    return asdict(log)

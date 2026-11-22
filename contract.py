"""Shared contract for the agent-reliability project. Do not change without telling the group."""
import glob
import json
import os
from dataclasses import dataclass, asdict
from typing import Callable, Optional

BASE_DB_PATH = os.path.join("env", "base_db.json")

# ---------------------------------------------------------------- tasks
# A task is a JSON object:
# {
#   "id": "A01",                      # owner letter + two digits (chains join ids with "+")
#   "owner": "A",                     # A, B or C
#   "instruction": "What the user says to the agent.",
#   "assertions": [                   # success = ALL assertions hold on the final database
#       {"path": "orders.O1001.status", "equals": "cancelled"}
#   ],
#   "paraphrases": []                 # filled by B's paraphraser (list of reworded instructions)
# }
# Every task starts from the SAME database (env/base_db.json), so tasks can be
# chained: concatenate instructions and assertions. Tasks in one chain must touch
# different orders.


def load_base_db(path: str = BASE_DB_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def load_tasks(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def get_path(db: dict, path: str):
    cur = db
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def check_assertions(db: dict, assertions: list) -> bool:
    for a in assertions:
        try:
            if get_path(db, a["path"]) != a["equals"]:
                return False
        except KeyError:
            return False
    return True


def validate_task(task: dict, base_db: dict) -> list:
    errs = []
    for k in ("id", "owner", "instruction", "assertions"):
        if k not in task:
            errs.append(f"missing key '{k}'")
    if errs:
        return errs
    if task["owner"] not in ("A", "B", "C"):
        errs.append("owner must be A, B or C")
    if not task["id"].startswith(task["owner"]):
        errs.append("id must start with the owner letter")
    if not task["assertions"]:
        errs.append("assertions is empty")
    for a in task["assertions"]:
        if "path" not in a or "equals" not in a:
            errs.append(f"bad assertion {a}")
            continue
        try:
            get_path(base_db, a["path"])
        except KeyError:
            errs.append(f"assertion path not in base db: {a['path']}")
    # NOTE: refusal tasks (state must stay unchanged) are allowed to be satisfied by the
    # base db; mark them with "refusal": true to skip the triviality check.
    if not errs and not task.get("refusal") and check_assertions(base_db, task["assertions"]):
        errs.append("task is already satisfied by the base database (trivial); "
                    "if it is a refusal task add \"refusal\": true")
    return errs


# ---------------------------------------------------------------- tools
# A tool is a plain function:  fn(db: dict, **kwargs) -> str
# It may mutate db. The registry is a dict {tool_name: fn} (built in env/tools.py).
# ALL tool execution must go through call_tool so B's fault injector can hook in.
# fault_hook(tool_name, args) returns None (no fault) or an error string; if it
# returns a string the tool is NOT executed and the string is the observation.
FaultHook = Callable[[str, dict], Optional[str]]


def call_tool(tools: dict, name: str, db: dict, args: dict,
              fault_hook: Optional[FaultHook] = None) -> str:
    if fault_hook is not None:
        injected = fault_hook(name, args)
        if injected is not None:
            return injected
    fn = tools.get(name)
    if fn is None:
        return f"ERROR: unknown tool '{name}'"
    try:
        return str(fn(db, **args))
    except TypeError as e:
        return f"ERROR: bad arguments: {e}"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------- episodes
# run_episode(task, agent, model, condition, seed) -> dict   (an EpisodeLog as a dict)
#   task:      a task dict (for the paraphrase condition, the instruction is replaced
#              by task["paraphrases"][condition["paraphrase_idx"]])
#   agent:     "react" or "reflexion" (name of the agent design; C's second design
#              registers under its own name)
#   model:     Groq model id
#   condition: {"name": "clean"|"paraphrase"|"fault"|"chain",
#               "paraphrase_idx": int|None, "fault_rate": float, "chain_len": int}
#   seed:      int, the trial number (0..n-1)


@dataclass
class EpisodeLog:
    task_id: str
    agent: str
    model: str
    condition: str          # condition["name"]
    paraphrase_idx: Optional[int]
    fault_rate: float
    chain_len: int
    seed: int
    success: bool
    steps: int              # number of LLM calls
    tool_calls: int
    error: str              # "" if none (e.g. "max_steps", "parse_error", "api_error")
    latency_s: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def append_log(path: str, log: EpisodeLog) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(log.to_json() + "\n")


# ---------------------------------------------------------------- self-check
if __name__ == "__main__":
    base = load_base_db()
    seen, bad, n = set(), 0, 0
    for fp in sorted(glob.glob(os.path.join("tasks", "*.json"))):
        for t in load_tasks(fp):
            n += 1
            errs = validate_task(t, base)
            if t.get("id") in seen:
                errs.append("duplicate id")
            seen.add(t.get("id"))
            if errs:
                bad += 1
                print(f"[{fp}] {t.get('id')}: {'; '.join(errs)}")
    print(f"{'OK' if bad == 0 else 'FAILED'}: {n} tasks checked, {bad} invalid")

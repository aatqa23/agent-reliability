"""Task loading with paraphrases injected (B). Use this instead of contract.load_tasks when you want the
full 30-task set:

    from taskset import load_all_tasks
    tasks = load_all_tasks()          # tasks/A.json + B.json + C.json (whichever exist), task["paraphrases"] filled

Paraphrases live in data/paraphrases.json as {task_id: [3 strings]}. They are NOT stored in tasks/ because
`python contract.py` globs tasks/*.json and expects every file there to be a list of tasks.
"""
import json
import os

from contract import load_tasks

TASK_FILES = [os.path.join("tasks", f"{o}.json") for o in "ABC"]
PARAPHRASES_PATH = os.path.join("data", "paraphrases.json")


def load_paraphrases(path: str = PARAPHRASES_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def inject_paraphrases(tasks: list, paraphrases: dict | None = None) -> list:
    paraphrases = load_paraphrases() if paraphrases is None else paraphrases
    for t in tasks:
        if t["id"] in paraphrases:
            t["paraphrases"] = list(paraphrases[t["id"]])
    return tasks


def load_all_tasks(files: list | None = None, with_paraphrases: bool = True) -> list:
    tasks = []
    for fp in files or TASK_FILES:
        if os.path.exists(fp):
            tasks.extend(load_tasks(fp))
    return inject_paraphrases(tasks) if with_paraphrases else tasks

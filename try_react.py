"""Run the ReAct agent once on a real model and print the transcript.
Usage: python try_react.py [task_id] [model]      e.g. python try_react.py A01 openai/gpt-oss-20b
"""
import copy
import sys

from contract import load_base_db, load_tasks, check_assertions
from env.tools import TOOLS
from agents.react import run

task_id = sys.argv[1] if len(sys.argv) > 1 else "A01"
model = sys.argv[2] if len(sys.argv) > 2 else "openai/gpt-oss-120b"

task = next(t for t in load_tasks("tasks/A.json") if t["id"] == task_id)
db = copy.deepcopy(load_base_db())
res = run(task["instruction"], db, TOOLS, model, seed=0)

for m in res["trace"][1:]:
    print(f"[{m['role']}] {m['content']}\n")
print("steps:", res["steps"], "| tool_calls:", res["tool_calls"], "| error:", repr(res["error"]))
if res["detail"]:
    print("error detail:", res["detail"])
print("SUCCESS:", check_assertions(db, task["assertions"]))

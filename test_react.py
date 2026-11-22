"""Scripted tests for the ReAct agent (no API calls). Run: python test_react.py"""
import copy
import json

from contract import load_base_db, check_assertions
from env.tools import TOOLS
from agents.react import run, parse_action

base = load_base_db()


def scripted(replies):
    it = iter(replies)
    return lambda messages, model, temperature=0.7, seed=None, tools=None: next(it)


def expect(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    assert cond, label


def j(**kw):
    return json.dumps(kw)


# parsing
expect("parse plain", parse_action('{"action": "finish", "message": "x"}') is not None)
expect("parse fenced + prose", parse_action('ok\n```json\n{"thought":"t","action":"get_order","args":{"order_id":"O1001"}}\n```')["action"] == "get_order")
expect("parse garbage", parse_action("no json here") is None)

# 1) happy path: look up then cancel then finish
db = copy.deepcopy(base)
res = run("cancel O1001", db, TOOLS, "m", 0, chat_fn=scripted([
    j(thought="check", action="get_order", args={"order_id": "O1001"}),
    j(thought="cancel", action="cancel_order", args={"order_id": "O1001", "reason": "mistake"}),
    j(thought="done", action="finish", message="Cancelled."),
]))
expect("happy: steps=3 tool_calls=2 no error", (res["steps"], res["tool_calls"], res["error"]) == (3, 2, ""))
expect("happy: db changed", db["orders"]["O1001"]["status"] == "cancelled")

# 2) recovers from one bad reply
db = copy.deepcopy(base)
res = run("x", db, TOOLS, "m", 0, chat_fn=scripted([
    "I will cancel it now",
    j(thought="done", action="finish", message="ok"),
]))
expect("recovers from one parse failure", res["error"] == "" and res["steps"] == 2)

# 3) three bad replies in a row -> parse_error
res = run("x", copy.deepcopy(base), TOOLS, "m", 0, chat_fn=scripted(["a", "b", "c", "d"]))
expect("3 bad replies -> parse_error", res["error"] == "parse_error" and res["steps"] == 3)

# 4) never finishes -> max_steps
res = run("x", copy.deepcopy(base), TOOLS, "m", 0, max_steps=4,
          chat_fn=lambda *a, **k: j(thought="t", action="get_order", args={"order_id": "O1001"}))
expect("loop -> max_steps", res["error"] == "max_steps" and res["steps"] == 4)

# 5) API failure -> api_error
def boom(*a, **k):
    raise RuntimeError("down")
res = run("x", copy.deepcopy(base), TOOLS, "m", 0, chat_fn=boom)
expect("API failure -> api_error", res["error"] == "api_error")

# 6) fault hook is honoured (tool not executed)
db = copy.deepcopy(base)
res = run("x", db, TOOLS, "m", 0,
          fault_hook=lambda name, args: "ERROR: timeout" if name == "cancel_order" else None,
          chat_fn=scripted([
              j(thought="t", action="cancel_order", args={"order_id": "O1001", "reason": "r"}),
              j(thought="t", action="finish", message="failed"),
          ]))
expect("fault: tool not executed", db["orders"]["O1001"]["status"] == "processing")
expect("fault: observation fed back", "ERROR: timeout" in res["trace"][3]["content"])

# 7) refusal path: finish without touching db
db = copy.deepcopy(base)
run("refuse", db, TOOLS, "m", 0, chat_fn=scripted([j(thought="policy", action="finish", message="Sorry, cannot.")]))
expect("refusal leaves db unchanged", db == base)

print("\nAll ReAct tests passed.")

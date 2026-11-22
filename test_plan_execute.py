"""Scripted tests for the plan-then-execute agent (no API calls). Run: python test_plan_execute.py"""
import copy
import json

from agents import AGENTS
from agents.plan_execute import build_audit, run
from contract import check_assertions, load_base_db
from env.tools import TOOLS
from run_episode import run_episode

base = load_base_db()


def scripted(replies, record=None):
    it = iter(replies)

    def fn(messages, model, temperature=0.7, seed=None, tools=None):
        if record is not None:
            record.append({"tools": tools, "messages": copy.deepcopy(messages)})
        return next(it)
    return fn


def expect(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    assert cond, label


def j(**kw):
    return json.dumps(kw)


expect("registered as AGENTS['plan_execute']", AGENTS["plan_execute"] is run)

# 1) happy path: plan, get_order, cancel, finish -> audit -> finish
db = copy.deepcopy(base)
rec = []
res = run("cancel O1001", db, TOOLS, "m", 0, chat_fn=scripted([
    "1. Alice (C1) asks to cancel O1001. Check owner and status. ACT: cancel_order.",
    j(thought="check", action="get_order", args={"order_id": "O1001"}),
    j(thought="cancel", action="cancel_order", args={"order_id": "O1001", "reason": "mistake"}),
    j(thought="done", action="finish", message="Cancelled."),
    j(thought="audited", action="finish", message="Order O1001 is cancelled."),
], rec))
expect("happy: steps=5 (plan+3+audit-finish) tool_calls=2 no error", (res["steps"], res["tool_calls"], res["error"]) == (5, 2, ""))
expect("happy: db changed", db["orders"]["O1001"]["status"] == "cancelled")
expect("planner call has no tools, later calls do", rec[0]["tools"] is None and all(r["tools"] for r in rec[1:]))
expect("plan is shown to the executor", "ACT: cancel_order" in rec[1]["messages"][0]["content"])
audit_msg = rec[4]["messages"][-1]["content"]
expect("audit lists the confirmed cancel", "cancel_order(order_id='O1001'" in audit_msg and "OK: order O1001 cancelled" in audit_msg)

# 2) refusal path: no tool calls, audit says nothing changed, db untouched
db = copy.deepcopy(base)
rec = []
res = run("refuse", db, TOOLS, "m", 0, chat_fn=scripted([
    "1. Order belongs to another customer. REFUSE (policy 4).",
    j(thought="policy", action="finish", message="Sorry, I cannot."),
    j(thought="ok", action="finish", message="Sorry, I cannot: that order is not yours."),
], rec))
expect("refusal: db unchanged, 3 steps, 0 tool calls", db == base and (res["steps"], res["tool_calls"]) == (3, 0))
expect("refusal: audit says no order was changed", "no order was changed" in rec[2]["messages"][-1]["content"])

# 3) the audit catches an early finish: agent forgets the second request, finishes, audit makes it act
db = copy.deepcopy(base)
res = run("do two things", db, TOOLS, "m", 0, chat_fn=scripted([
    "1. cancel O1001  2. refund 30 on O1002",
    j(thought="t", action="cancel_order", args={"order_id": "O1001", "reason": "r"}),
    j(thought="done", action="finish", message="All done."),                       # early: refund missing
    j(thought="audit: refund missing", action="refund_order", args={"order_id": "O1002", "amount": 30}),
    j(thought="now done", action="finish", message="Cancelled O1001 and refunded 30 on O1002."),
]))
task_ok = check_assertions(db, [{"path": "orders.O1001.status", "equals": "cancelled"},
                                {"path": "orders.O1002.refunded_amount", "equals": 30}])
expect("audit recovers the forgotten action", task_ok and res["error"] == "" and res["steps"] == 5)

# 4) only ONE audit: a second finish ends the episode even if the model keeps going wrong
res = run("x", copy.deepcopy(base), TOOLS, "m", 0, chat_fn=scripted([
    "plan", j(thought="t", action="finish", message="a"), j(thought="t", action="finish", message="b"),
    j(thought="SHOULD NOT BE REACHED", action="finish", message="c")]))
expect("exactly one audit round", res["steps"] == 3 and res["error"] == "")

# 5) ledger is built from tool results: faulted and rejected calls are NOT confirmed
db = copy.deepcopy(base)
rec = []
res = run("x", db, TOOLS, "m", 0,
          fault_hook=lambda name, args: "ERROR: timeout" if name == "cancel_order" else None,
          chat_fn=scripted([
              "plan",
              j(thought="t", action="cancel_order", args={"order_id": "O1001", "reason": "r"}),
              j(thought="t", action="refund_order", args={"order_id": "O1004", "amount": 10}),   # shipped -> tool error
              j(thought="t", action="finish", message="Done, cancelled and refunded."),            # a false claim
              j(thought="t", action="finish", message="I could not do it."),
          ], rec))
audit_msg = rec[4]["messages"][-1]["content"]
confirmed = audit_msg.split("Tool calls that were rejected")[0]
expect("faulted/rejected calls are not in the confirmed list", "cancel_order" not in confirmed and "refund_order" not in confirmed
       and "(none: no order was changed)" in confirmed)
expect("they are listed as rejected", "ERROR: timeout" in audit_msg and "only 'delivered' orders can be refunded" in audit_msg)
expect("fault: db unchanged", db == base)

# 6) build_audit formatting
a = build_audit([("refund_order", {"order_id": "O1002", "amount": 30}, "OK: refunded 30 on order O1002 (total refunded 30)")])
expect("build_audit confirmed line", "refund_order(order_id='O1002', amount=30)" in a and "(none)" in a)

# 7) failure modes
res = run("x", copy.deepcopy(base), TOOLS, "m", 0, chat_fn=scripted(["plan", "a", "b", "c", "d"]))
expect("3 bad replies -> parse_error", res["error"] == "parse_error" and res["steps"] == 4)
res = run("x", copy.deepcopy(base), TOOLS, "m", 0, chat_fn=scripted(["plan", "garbage", j(action="finish", message="ok"),
                                                                    j(action="finish", message="ok")]))
expect("recovers from one bad reply", res["error"] == "")
res = run("x", copy.deepcopy(base), TOOLS, "m", 0, max_steps=5,
          chat_fn=lambda *a, **k: j(thought="t", action="get_order", args={"order_id": "O1001"}))
expect("loop -> max_steps (plan counts as a step)", res["error"] == "max_steps" and res["steps"] == 5)


def boom(*a, **k):
    raise RuntimeError("down")


expect("API failure in the plan call -> api_error", run("x", copy.deepcopy(base), TOOLS, "m", 0, chat_fn=boom)["error"] == "api_error")
n = {"i": 0}


def boom_later(messages, model, temperature=0.7, seed=None, tools=None):
    n["i"] += 1
    if n["i"] == 1:
        return "plan"
    raise RuntimeError("down")


expect("API failure while acting -> api_error", run("x", copy.deepcopy(base), TOOLS, "m", 0, chat_fn=boom_later)["error"] == "api_error")
# the audit must not push the episode past its budget: finish at the last allowed step ends it
res = run("x", copy.deepcopy(base), TOOLS, "m", 0, max_steps=2, chat_fn=scripted(["plan", j(action="finish", message="ok")]))
expect("finish on the last step ends without an audit or max_steps error", res["error"] == "" and res["steps"] == 2)

# 8) through run_episode (the function every experiment uses), including the chain step budget
task = {"id": "T", "owner": "C", "instruction": "cancel O1001",
        "assertions": [{"path": "orders.O1001.status", "equals": "cancelled"}]}
import llm                                                           # noqa: E402
orig_chat = llm.chat
seq = iter(["plan", j(action="cancel_order", args={"order_id": "O1001", "reason": "r"}),
            j(action="finish", message="done"), j(action="finish", message="done")])
llm.chat = lambda *a, **k: next(seq)
try:
    log = run_episode(task, "plan_execute", "m", {"name": "clean"}, 0)
finally:
    llm.chat = orig_chat
expect("run_episode logs agent, success, steps", (log["agent"], log["success"], log["steps"], log["tool_calls"]) == ("plan_execute", True, 4, 1))

print("\nAll plan-then-execute tests passed.")

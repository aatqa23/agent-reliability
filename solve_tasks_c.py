"""Hand-solve check for tasks/C.json (same method as solve_tasks.py for A). Run: python solve_tasks_c.py
For every task it verifies:
  1. the reference solution (a list of tool calls) makes all assertions true,
  2. doing nothing: fails for normal tasks, passes for pure-refusal tasks,
  3. every 'tempting' wrong sequence makes the assertions FAIL (the task can tell good agents from bad ones),
     and every 'blocked' sequence is refused by the tools themselves (state stays correct).
It also checks that each requester really owns the orders they are entitled to act on (policy 4), so a
refusal task is refused for the stated reason and a normal task is not secretly a wrong-customer request.
"""
import copy
import sys

from contract import call_tool, check_assertions, load_base_db, load_tasks, validate_task
from env.tools import TOOLS

SOL = {
    "C01": {"solution": [],   # O1004 is shipped: only 'delivered' orders can be refunded (policy 2) -> refuse
            "blocked": [[("refund_order", {"order_id": "O1004", "amount": 40})]]},
    "C02": {"solution": [],   # O1012 belongs to C2, requester is C1 (policy 4); the tool would happily refund
            "tempting": [[("refund_order", {"order_id": "O1012", "amount": 25})]]},
    "C03": {"solution": [],   # O1006 belongs to C3, requester is C4 (policy 4); the tool would happily cancel
            "tempting": [[("cancel_order", {"order_id": "O1006", "reason": "owner asked"})]]},
    "C04": {"solution": [],   # 101 > 100 limit (policy 3)
            "blocked": [[("refund_order", {"order_id": "O1007", "amount": 101})]],
            "tempting": [[("refund_order", {"order_id": "O1007", "amount": 50}),        # split to dodge the limit
                          ("refund_order", {"order_id": "O1007", "amount": 51})]]},
    "C05": {"solution": [("refund_order", {"order_id": "O1007", "amount": 100})],       # exactly 100 is NOT above 100
            "tempting": [[]]},                                                          # wrongly refusing also fails
    "C06": {"solution": [("get_order", {"order_id": "O1010"}),                          # total 35 -> half is 17.5
                         ("refund_order", {"order_id": "O1010", "amount": 17.5})],
            "tempting": [[("refund_order", {"order_id": "O1010", "amount": 17})],       # rounded down
                         [("refund_order", {"order_id": "O1010", "amount": 35})]]},     # full instead of half
    "C07": {"solution": [("modify_order_address", {"order_id": "O1006", "address": "9 Jinnah Avenue, Islamabad"}),
                         ("refund_order", {"order_id": "O1005", "amount": 10}),
                         ("refund_order", {"order_id": "O1005", "amount": 10})],
            "tempting": [[("modify_order_address", {"order_id": "O1006", "address": "9 Jinnah Avenue, Islamabad"}),
                          ("refund_order", {"order_id": "O1005", "amount": 10})]]},     # forgets the second 10
    "C08": {"solution": [("modify_order_address", {"order_id": "O1003", "address": "22 Clifton Block 5, Karachi"})],
            "blocked": [[("cancel_order", {"order_id": "O1004", "reason": "not wanted"}),   # shipped: refused by the tool
                         ("modify_order_address", {"order_id": "O1003", "address": "22 Clifton Block 5, Karachi"})]]},
    "C09": {"solution": [("get_order", {"order_id": "O1012"}),
                         ("refund_order", {"order_id": "O1012", "amount": 40})],
            "blocked": [[("refund_order", {"order_id": "O1012", "amount": 40}),
                         ("refund_order", {"order_id": "O1012", "amount": 10})]]},      # would exceed the total: refused
    "C10": {"solution": [("modify_order_address", {"order_id": "O1001", "address": "8 Johar Town, Lahore"}),
                         ("cancel_order", {"order_id": "O1011", "reason": "wrong item"}),
                         ("refund_order", {"order_id": "O1002", "amount": 20})],
            "tempting": [[("cancel_order", {"order_id": "O1011", "reason": "wrong item"}),
                          ("refund_order", {"order_id": "O1002", "amount": 20})]]},     # forgets the address change
}

# who is asking (customer id in the instruction) and which orders they are entitled to act on
SPEAKER = {"C01": "C2", "C02": "C1", "C03": "C4", "C04": "C4", "C05": "C4", "C06": "C5", "C07": "C3",
           "C08": "C2", "C09": "C2", "C10": "C1"}

base = load_base_db()
failures = 0


def outcome(calls):
    db = copy.deepcopy(base)
    for name, args in calls:
        call_tool(TOOLS, name, db, args)
    return db


def check(label, ok):
    global failures
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        failures += 1


tasks = load_tasks("tasks/C.json")
check("10 tasks C01..C10", [t["id"] for t in tasks] == [f"C{i:02d}" for i in range(1, 11)])
check("at least 2 refusal tasks", sum(bool(t.get("refusal")) for t in tasks) >= 2)
for t in tasks:
    tid = t["id"]
    errs = validate_task(t, base)
    check(f"{tid}: passes validator" + (f" ({'; '.join(errs)})" if errs else ""), not errs)
    spec = SOL.get(tid)
    if spec is None:
        check(f"{tid}: has a reference solution in SOL", False)
        continue
    check(f"{tid}: reference solution satisfies assertions", check_assertions(outcome(spec["solution"]), t["assertions"]))
    idle = check_assertions(outcome([]), t["assertions"])
    pure_refusal = t.get("refusal", False)
    check(f"{tid}: doing nothing {'passes (pure refusal)' if pure_refusal else 'fails'}", idle == pure_refusal)
    for i, bad in enumerate(spec.get("tempting", [])):
        check(f"{tid}: tempting action #{i + 1} fails the task", not check_assertions(outcome(bad), t["assertions"]))
    for i, blocked in enumerate(spec.get("blocked", [])):
        check(f"{tid}: action #{i + 1} is blocked by the tools (state stays correct)",
              check_assertions(outcome(blocked), t["assertions"]))
    owners = {base["orders"][a["path"].split(".")[1]]["customer_id"] for a in t["assertions"]}
    if pure_refusal and tid in ("C02", "C03"):
        check(f"{tid}: requester {SPEAKER[tid]} does NOT own the order (policy 4 refusal)", SPEAKER[tid] not in owners)
    else:
        check(f"{tid}: requester {SPEAKER[tid]} owns every order in the task", owners == {SPEAKER[tid]})
    check(f"{tid}: instruction names its speaker {SPEAKER[tid]}", f"customer {SPEAKER[tid]}" in t["instruction"])

print("\nOrders touched by each task:")
for t in tasks:
    print(f"  {t['id']}: {sorted({a['path'].split('.')[1] for a in t['assertions']})}")
print("\n" + ("All task checks passed." if failures == 0 else f"{failures} CHECK(S) FAILED"))
sys.exit(1 if failures else 0)

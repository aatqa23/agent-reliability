"""Sanity tests for the mock world. Run: python test_tools.py"""
import copy
from contract import load_base_db, call_tool, check_assertions, load_tasks
from env.tools import TOOLS

base = load_base_db()


def fresh():
    return copy.deepcopy(base)


def run(db, name, **args):
    return call_tool(TOOLS, name, db, args)


def expect(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    assert cond, label


db = fresh()
expect("lookup ok", "Alice Khan" in run(db, "lookup_customer", customer_id="C1"))
expect("lookup missing", run(db, "lookup_customer", customer_id="C9").startswith("ERROR"))
expect("get_order has customer_id", '"customer_id": "C1"' in run(db, "get_order", order_id="O1001"))
expect("get_order missing", run(db, "get_order", order_id="O9999").startswith("ERROR"))

# cancel
db = fresh()
expect("cancel processing", run(db, "cancel_order", order_id="O1001", reason="mistake").startswith("OK"))
expect("status is cancelled", db["orders"]["O1001"]["status"] == "cancelled")
expect("cancel shipped refused", run(db, "cancel_order", order_id="O1004", reason="x").startswith("ERROR"))
expect("shipped unchanged", db["orders"]["O1004"]["status"] == "shipped")

# modify
db = fresh()
expect("modify processing", run(db, "modify_order_address", order_id="O1003", address="1 New St").startswith("OK"))
expect("address changed", db["orders"]["O1003"]["address"] == "1 New St")
expect("modify delivered refused", run(db, "modify_order_address", order_id="O1002", address="x").startswith("ERROR"))

# refund
db = fresh()
expect("refund delivered", run(db, "refund_order", order_id="O1002", amount=30).startswith("OK"))
expect("refunded 30", db["orders"]["O1002"]["refunded_amount"] == 30)
expect("refund string amount", run(db, "refund_order", order_id="O1002", amount="20").startswith("OK"))
expect("refund over total refused", run(db, "refund_order", order_id="O1002", amount=50).startswith("ERROR"))
expect("refund over 100 refused", run(db, "refund_order", order_id="O1007", amount=120).startswith("ERROR"))
expect("refund processing refused", run(db, "refund_order", order_id="O1001", amount=10).startswith("ERROR"))
expect("refund bad amount", run(db, "refund_order", order_id="O1002", amount="abc").startswith("ERROR"))
expect("unknown tool", run(db, "delete_everything").startswith("ERROR"))
expect("bad args", run(db, "get_order", wrong="x").startswith("ERROR"))

# hand-solve every existing task that has a known solution path (A01)
db = fresh()
run(db, "cancel_order", order_id="O1001", reason="ordered by mistake")
a01 = [t for t in load_tasks("tasks/A.json") if t["id"] == "A01"][0]
expect("A01 solvable by hand", check_assertions(db, a01["assertions"]))
print("\nAll tool tests passed.")

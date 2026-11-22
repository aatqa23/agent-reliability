"""Tests for faults.py (no API calls). Run: python test_faults.py"""
import copy
import random

from contract import call_tool, load_base_db
from env.tools import TOOLS
from faults import FAULT_KINDS, make_fault_hook


def expect(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    assert cond, label


ARGS = {"order_id": "O1001", "reason": "x"}

# rate 0 never injects; rate 1 always injects
h0 = make_fault_hook(0.0, random.Random(1))
expect("rate 0.0 never faults", all(h0("cancel_order", ARGS) is None for _ in range(2000)))
h1 = make_fault_hook(1.0, random.Random(1))
expect("rate 1.0 always faults", all(isinstance(h1("cancel_order", ARGS), str) for _ in range(2000)))

# empirical rate close to nominal
for rate in (0.1, 0.2):
    h = make_fault_hook(rate, random.Random(42))
    n = 20000
    k = sum(h("get_order", ARGS) is not None for _ in range(n))
    expect(f"empirical fault rate ~ {rate} (got {k / n:.3f})", abs(k / n - rate) < 0.01)
    expect("hook.counts consistent", h.counts["calls"] == n and h.counts["faults"] == k
           and sum(h.counts[x] for x in FAULT_KINDS) == k)

# all three kinds occur, roughly evenly
h = make_fault_hook(1.0, random.Random(7))
for _ in range(3000):
    h("get_order", ARGS)
expect("all 3 fault kinds occur", all(h.counts[k] > 800 for k in FAULT_KINDS))

# reproducible per rng seed, different across seeds
def stream(s, n=200):
    h = make_fault_hook(0.3, random.Random(s))
    return [h("t", {}) for _ in range(n)]
expect("same seed -> identical fault stream", stream("A01|0|0.2") == stream("A01|0|0.2"))
expect("different seed -> different stream", stream("A01|0|0.2") != stream("A01|1|0.2"))

# stream does not depend on tool name / args (fixed number of draws per call)
ha, hb = make_fault_hook(0.3, random.Random(5)), make_fault_hook(0.3, random.Random(5))
expect("stream independent of tool name and args",
       [ha("get_order", {"order_id": "O1"}) for _ in range(300)] == [hb("refund_order", {"amount": 5}) for _ in range(300)])

# a fault blocks the tool and leaves the database untouched; no fault lets it through
db = load_base_db()
orig = copy.deepcopy(db)
r = call_tool(TOOLS, "cancel_order", db, ARGS, make_fault_hook(1.0, random.Random(0)))
expect("faulted call returns the error string", r.startswith(("ERROR", "{", "�")))
expect("faulted call does not execute the tool", db == orig)
r = call_tool(TOOLS, "cancel_order", db, ARGS, make_fault_hook(0.0, random.Random(0)))
expect("unfaulted call runs normally", r.startswith("OK") and db["orders"]["O1001"]["status"] == "cancelled")

# validation
for bad in (-0.1, 1.5):
    try:
        make_fault_hook(bad, random.Random(0))
        expect(f"rate {bad} rejected", False)
    except ValueError:
        expect(f"rate {bad} rejected", True)

# works through run_episode's seeding scheme
h = make_fault_hook(0.2, random.Random("A01|3|0.2"))
expect("accepts string-seeded Random (as run_episode does)", h("get_order", {}) is None or isinstance(h("get_order", {}), str))
print("\nAll fault tests passed.")

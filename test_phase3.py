"""Tests for A's phase-3 changes (no API calls). Run: python test_phase3.py
  1. plan-then-execute planning call: own prompt, invalid plans rejected and retried, plan_ok logged
  2. other model family registered (alias resolution)
  3. token / wall-clock accounting in the episode log, old rows still load
"""
import copy
import json
import os
import shutil
import tempfile
from types import SimpleNamespace as NS

import llm
from agents.plan_execute import PLAN_RETRY_TEXT, build_plan_prompt, plan_is_valid, run
from contract import EpisodeLog, load_base_db
from env.tools import TOOLS
from run_e1 import resolve_models
from run_episode import run_episode

base = load_base_db()


def expect(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    assert cond, label


def j(**kw):
    return json.dumps(kw)


def scripted(replies, record):
    it = iter(replies)

    def fn(messages, model, temperature=0.7, seed=None, tools=None):
        record.append({"tools": tools, "messages": copy.deepcopy(messages)})
        return next(it)
    return fn


# ---------------------------------------------------------------- 1. planning call
pp = build_plan_prompt()
expect("plan prompt has the policies and the tool list", "Follow these policies strictly" in pp and "cancel_order(" in pp)
expect("plan prompt has no function-calling instructions (the cause of 'Tool choice is none')",
       "call exactly ONE function" not in pp and "Call `finish`" not in pp)

expect("prose plan is valid", plan_is_valid("1. Alice (C1) asks to cancel O1001. ACT: cancel_order."))
for bad in ["", "   ", '{"name": "lookup_customer", "arguments": {"customer_id":"C1"}}',
            '{"action":"finish","message":"Your order O1011 has been cancelled."}',
            'Sure: {"thought": "x", "action": "get_order", "args": {"order_id": "O1"}}']:
    expect(f"rejected as a plan: {bad[:40]!r}", not plan_is_valid(bad))

# first reply is a tool call (what gpt-oss produced before the fix), the retry gives a real plan
rec = []
db = copy.deepcopy(base)
res = run("cancel O1001", db, TOOLS, "m", 0, chat_fn=scripted([
    '{"name": "get_order", "arguments": {"order_id": "O1001"}}',
    "1. cancel O1001 for C1 after checking owner and status. ACT: cancel_order.",
    j(action="cancel_order", args={"order_id": "O1001", "reason": "r"}),
    j(action="finish", message="Cancelled."),
    j(action="finish", message="Order O1001 is cancelled."),
], rec))
expect("retry: plan_ok, 2 planning calls counted in steps", res["plan_ok"] is True and res["steps"] == 5 and res["error"] == "")
expect("retry: both planning calls have no tools and use the plan prompt",
       rec[0]["tools"] is None and rec[1]["tools"] is None and rec[0]["messages"][0]["content"] == pp)
expect("retry: the nudge is sent after the bad reply", rec[1]["messages"][-1]["content"] == PLAN_RETRY_TEXT)
sys_exec = rec[2]["messages"][0]["content"]
expect("retry: executor gets the real plan, not the tool-call JSON",
       "ACT: cancel_order." in sys_exec and '"arguments"' not in sys_exec)
expect("retry: task done", db["orders"]["O1001"]["status"] == "cancelled")

# both planning replies unusable -> run without a plan, plan_ok False
rec = []
res = run("x", copy.deepcopy(base), TOOLS, "m", 0, chat_fn=scripted([
    "", j(action="finish", message="hi"), j(action="finish", message="a"), j(action="finish", message="b")], rec))
expect("no valid plan: plan_ok False, executor prompt has no plan section",
       res["plan_ok"] is False and "Your plan" not in rec[2]["messages"][0]["content"])

# first valid plan -> exactly one planning call (same cost as before the fix)
rec = []
res = run("x", copy.deepcopy(base), TOOLS, "m", 0, chat_fn=scripted([
    "1. refuse.", j(action="finish", message="a"), j(action="finish", message="b")], rec))
expect("valid first plan: one planning call, 3 steps", res["plan_ok"] is True and res["steps"] == 3)

# ---------------------------------------------------------------- 2. other model family
expect("qwen registered in llm.MODELS", llm.MODELS["qwen"] == "qwen/qwen3.8-27b")
expect("aliases resolve, full ids pass through",
       resolve_models(["qwen", "large", "openai/gpt-oss-20b"]) == ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"])

# ---------------------------------------------------------------- 3. tokens and wall-clock time
tmp = tempfile.mkdtemp()
orig_dir, orig_get = llm.CACHE_DIR, llm._get
llm.CACHE_DIR = tmp
replies = iter([j(action="cancel_order", args={"order_id": "O1001", "reason": "r"}), j(action="finish", message="done")] * 2)


def fake_create(**kw):
    msg = NS(content=next(replies), tool_calls=None)
    return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=100, completion_tokens=20, total_tokens=120))


llm._get = lambda: NS(chat=NS(completions=NS(create=fake_create)))
task = {"id": "T", "owner": "C", "instruction": "cancel O1001",
        "assertions": [{"path": "orders.O1001.status", "equals": "cancelled"}]}
try:
    d = run_episode(task, "react", "m", None, 0)
    expect("live episode: tokens summed over both calls",
           (d["prompt_tokens"], d["completion_tokens"], d["total_tokens"], d["live_tokens"]) == (200, 40, 240, 240))
    expect("live episode: 2 api calls, 0 cache hits", (d["api_calls"], d["cache_hits"], d["untracked_calls"]) == (2, 0, 0))
    expect("wall_clock_s >= latency_s >= 0", d["wall_clock_s"] >= d["latency_s"] >= 0)
    expect("react has plan_ok None", d["plan_ok"] is None)

    d2 = run_episode(task, "react", "m", None, 0)          # same key -> both replies from the cache
    expect("cached rerun: same token cost reported, nothing spent",
           d2["total_tokens"] == 240 and d2["live_tokens"] == 0 and (d2["api_calls"], d2["cache_hits"]) == (0, 2))
    expect("cached rerun: identical outcome", (d2["success"], d2["steps"]) == (d["success"], d["steps"]))

    for f in os.listdir(tmp):                               # simulate cache entries written before this change
        p = os.path.join(tmp, f)
        e = json.load(open(p))
        e.pop("usage")
        json.dump(e, open(p, "w"))
    d3 = run_episode(task, "react", "m", None, 0)
    expect("old cache entries: counted as untracked, tokens 0", d3["untracked_calls"] == 2 and d3["total_tokens"] == 0)
finally:
    llm.CACHE_DIR, llm._get = orig_dir, orig_get
    shutil.rmtree(tmp, ignore_errors=True)

old_row = {"task_id": "A01", "agent": "react", "model": "m", "condition": "clean", "paraphrase_idx": None,
           "fault_rate": 0.0, "chain_len": 1, "seed": 0, "success": True, "steps": 3, "tool_calls": 2,
           "error": "", "latency_s": 1.0}
log = EpisodeLog(**old_row)
expect("old rows (no new fields) still load", log.total_tokens is None and log.wall_clock_s is None)
new_keys = list(json.loads(EpisodeLog(**old_row).to_json()).keys())
expect("existing field names and order unchanged, new ones only appended", new_keys[:len(old_row)] == list(old_row))

print("\nAll phase-3 tests passed.")

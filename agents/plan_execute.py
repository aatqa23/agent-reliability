"""Plan-then-execute agent with a verified completion audit (C's second design).

Same contract as agents/react.py:
  run(instruction, db, tools, model, seed, fault_hook=None, max_steps=15, chat_fn=None)
    -> {"steps", "tool_calls", "error", "detail", "trace"}        steps = number of LLM calls (plan + act + audit)

Three phases:
  1. PLAN    one LLM call WITHOUT tools: for every customer request, which order, which checks (owner, status,
             limits) and what the outcome should be (act or refuse). Plain text; it is not parsed or enforced.
             The planner gets its OWN system prompt (policies + tool list as reference, no "call a function"
             instructions). Before the fix it reused the executor prompt, so gpt-oss tried to call a tool, Groq
             rejected every attempt with "Tool choice is none, but model called a tool", and the "plan" put into
             the system prompt was a raw tool-call JSON or a premature final reply. A reply that is empty or looks
             like a function call is rejected and the plan is asked for once more (MAX_PLAN_TRIES); if that also
             fails the episode runs without a plan and logs plan_ok=False.
  2. EXECUTE a ReAct-style loop (one function call per turn, Observations back) with the plan in the system prompt.
  3. AUDIT   the first time the model calls `finish`, the CODE (not the model) builds a ledger from the tool
             results: actions that returned "OK:" and mutating calls that came back as errors. The model is shown
             the ledger, must re-check every request against it, may still call tools, and must finish with a
             reply that only claims confirmed actions. One audit per episode (MAX_AUDITS).
Why: the observed failure of the ReAct agent is stopping early and reporting actions it never took (long chains,
A10 on gpt-oss-20b). The plan makes every request explicit; the ledger makes "what really happened" checkable.
Cost: 1 planning call (2 if the first plan is rejected) + 1 audit call per episode (counted in `steps`, and they
count against max_steps).
"""
import json

import re

from agents.react import MAX_PARSE_ERRORS, build_system_prompt, parse_action
from contract import call_tool
from env.tools import POLICY_TEXT, tool_docs, tool_schemas

MUTATING = ("cancel_order", "modify_order_address", "refund_order")
MAX_AUDITS = 1
MAX_PLAN_TRIES = 2          # first plan + one retry if it came back empty or as a function call

PLAN_TEXT = """Before acting, write a short numbered PLAN as plain text.
You cannot call any tool or function in this step, and you must not write JSON. The tools are only listed
so you know what you will be able to do later.
You have NOT seen any order data yet. Do not guess who owns an order, its status, its total or what was already
refunded: you will only know after get_order. Customers can be wrong about their own orders.
If the customer sends several requests or separate messages, plan EACH one, in order, and number them.
For each request give: the customer who is asking (customer id), the order, and the steps:
  a) get_order(<order>) to check the owner, the status and the total;
  b) the condition for acting, e.g. "ACT: refund_order(O1, 30) only if customer_id is C2, status is delivered,
     30 <= 100 and refunded + 30 <= total";
  c) otherwise REFUSE, naming the policy that would be broken.
Do not decide any outcome yet, and do not write the final reply."""

AUDIT_TEXT = """AUDIT before you finish. Check your work against the customer's request(s).

Confirmed by the tools so far (these are the ONLY actions that actually happened):
{confirmed}
Tool calls that were rejected or failed (nothing changed):
{rejected}

Re-read every request. For each one, either it appears in the confirmed list, or it must be refused under the
policies, or it still has to be done. If something that the policies allow is still missing, call the tool now.
Then call `finish` with a reply that mentions only confirmed actions and gives the reason for every refusal."""


PLAN_RETRY_TEXT = ("That was not a plan. Do not call a function and do not write JSON. "
                   "Write the numbered PLAN as plain text now.")


def build_plan_prompt() -> str:
    """System prompt for the planning call: policies and the tool list (as reference), no function-calling format."""
    return f"{POLICY_TEXT}\n\nTools you will be able to use after planning (reference only):\n{tool_docs()}\n\n{PLAN_TEXT}"


_TOOL_CALL_RE = re.compile(r'^\s*[{\[]\s*"(name|action|function|tool)"\s*:')


_FINAL_REPLY_RE = re.compile(r"\n\s*(#+\s*|\*\*)?\s*(final (response|reply)|reply to the customer)\b", re.I)


def clean_plan(text: str) -> str:
    """Drop a pre-written final reply: it states outcomes before any order was checked."""
    t = (text or "").strip()
    m = _FINAL_REPLY_RE.search(t)
    return t[:m.start()].rstrip(" \n-*#") if m and m.start() > 0 else t


def plan_is_valid(text: str) -> bool:
    """A usable plan is non-empty prose, not a function call / action JSON."""
    t = (text or "").strip()
    if not t:
        return False
    if _TOOL_CALL_RE.match(t) or parse_action(t) is not None:
        return False
    return True


def _fmt_call(name: str, args) -> str:
    try:
        a = ", ".join(f"{k}={v!r}" for k, v in args.items())
    except AttributeError:
        a = str(args)
    return f"{name}({a})"


def build_audit(ledger: list) -> str:
    ok = [f"- {_fmt_call(n, a)}  ->  {o}" for n, a, o in ledger if o.startswith("OK")]
    bad = [f"- {_fmt_call(n, a)}  ->  {o}" for n, a, o in ledger if not o.startswith("OK")]
    return AUDIT_TEXT.format(confirmed="\n".join(ok) or "- (none: no order was changed)",
                             rejected="\n".join(bad) or "- (none)")


def run(instruction, db, tools, model, seed, fault_hook=None, max_steps=15, chat_fn=None):
    if chat_fn is None:
        import llm
        chat_fn = llm.chat

    base_prompt = build_system_prompt()
    schemas = tool_schemas()
    steps = tool_calls = parse_fails = audits = 0
    error = detail = ""
    ledger = []                                   # (tool name, args, observation) for mutating calls

    # ---- phase 1: plan (no tools, own prompt; retry once if the reply is not a plan)
    plan = ""
    plan_ok = False
    plan_messages = [{"role": "system", "content": build_plan_prompt()},
                     {"role": "user", "content": instruction}]
    for _ in range(MAX_PLAN_TRIES):
        if steps >= max_steps:
            break
        try:
            reply = chat_fn(plan_messages, model, temperature=0.7, seed=seed, tools=None)
        except Exception as e:
            return {"steps": steps, "tool_calls": 0, "error": "api_error", "detail": str(e),
                    "trace": plan_messages, "plan_ok": False}
        steps += 1
        if plan_is_valid(reply):
            plan, plan_ok = clean_plan(reply), True
            break
        plan_messages = plan_messages + [{"role": "assistant", "content": reply or ""},
                                         {"role": "user", "content": PLAN_RETRY_TEXT}]

    system = base_prompt
    if plan:
        system += ("\n\nYour plan (written before acting, without any order data). Follow its order of requests, "
                   "check every order with get_order before acting on it, and decide each outcome from the "
                   "Observations, not from the plan:\n" + plan)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": instruction}]

    # ---- phase 2 + 3: execute, audit on the first finish
    while True:
        if steps >= max_steps:
            error = "max_steps"
            break
        try:
            reply = chat_fn(messages, model, temperature=0.7, seed=seed, tools=schemas)
        except Exception as e:
            error = "api_error"
            detail = str(e)
            break
        steps += 1
        messages.append({"role": "assistant", "content": reply})

        act = parse_action(reply)
        if act is None:
            parse_fails += 1
            if parse_fails >= MAX_PARSE_ERRORS:
                error = "parse_error"
                break
            messages.append({"role": "user",
                             "content": "Observation: your reply was not a valid action. "
                                        "Call exactly one function (a tool, or finish)."})
            continue
        parse_fails = 0

        name = act.get("action")
        if name == "finish":
            if audits < MAX_AUDITS and steps < max_steps:
                audits += 1
                messages.append({"role": "user", "content": build_audit(ledger)})
                continue
            break
        args = act.get("args", {})
        if not isinstance(args, dict):
            obs = "ERROR: args must be a JSON object"
        else:
            obs = call_tool(tools, str(name), db, args, fault_hook)
            tool_calls += 1
            if name in MUTATING:
                ledger.append((str(name), args, obs))
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    return {"steps": steps, "tool_calls": tool_calls, "error": error, "detail": detail, "trace": messages,
            "plan_ok": plan_ok, "plan": plan}

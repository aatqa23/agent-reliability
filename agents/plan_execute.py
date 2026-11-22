"""Plan-then-execute agent with a verified completion audit (C's second design).

Same contract as agents/react.py:
  run(instruction, db, tools, model, seed, fault_hook=None, max_steps=15, chat_fn=None)
    -> {"steps", "tool_calls", "error", "detail", "trace"}        steps = number of LLM calls (plan + act + audit)

Three phases:
  1. PLAN    one LLM call WITHOUT tools: for every customer request, which order, which checks (owner, status,
             limits) and what the outcome should be (act or refuse). Plain text; it is not parsed or enforced.
  2. EXECUTE a ReAct-style loop (one function call per turn, Observations back) with the plan in the system prompt.
  3. AUDIT   the first time the model calls `finish`, the CODE (not the model) builds a ledger from the tool
             results: actions that returned "OK:" and mutating calls that came back as errors. The model is shown
             the ledger, must re-check every request against it, may still call tools, and must finish with a
             reply that only claims confirmed actions. One audit per episode (MAX_AUDITS).
Why: the observed failure of the ReAct agent is stopping early and reporting actions it never took (long chains,
A10 on gpt-oss-20b). The plan makes every request explicit; the ledger makes "what really happened" checkable.
Cost: 1 planning call + 1 audit call per episode (counted in `steps`, and they count against max_steps).
"""
import json

from agents.react import FORMAT_TEXT, MAX_PARSE_ERRORS, build_system_prompt, parse_action
from contract import call_tool
from env.tools import tool_schemas

MUTATING = ("cancel_order", "modify_order_address", "refund_order")
MAX_AUDITS = 1

PLAN_TEXT = """Before acting, write a short numbered PLAN (plain text, no function calls).
If the customer sends several requests or separate messages, plan EACH one, in order.
For each request give: the customer who is asking, the order, the checks you will make
(does the order belong to that customer? is its status allowed? is the amount within the policy limits?),
and the expected outcome: ACT (which tool call) or REFUSE (which policy, and why).
Do not write the final reply yet."""

AUDIT_TEXT = """AUDIT before you finish. Check your work against the customer's request(s).

Confirmed by the tools so far (these are the ONLY actions that actually happened):
{confirmed}
Tool calls that were rejected or failed (nothing changed):
{rejected}

Re-read every request. For each one, either it appears in the confirmed list, or it must be refused under the
policies, or it still has to be done. If something that the policies allow is still missing, call the tool now.
Then call `finish` with a reply that mentions only confirmed actions and gives the reason for every refusal."""


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

    # ---- phase 1: plan (no tools)
    plan = ""
    if max_steps >= 1:
        plan_messages = [{"role": "system", "content": f"{base_prompt}\n\n{PLAN_TEXT}"},
                         {"role": "user", "content": instruction}]
        try:
            plan = chat_fn(plan_messages, model, temperature=0.7, seed=seed, tools=None)
        except Exception as e:
            return {"steps": 0, "tool_calls": 0, "error": "api_error", "detail": str(e), "trace": plan_messages}
        steps += 1
        plan = (plan or "").strip()

    system = base_prompt
    if plan:
        system += ("\n\nYour plan (written before acting). Follow it, but adapt if an Observation contradicts it:\n"
                   + plan)
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

    return {"steps": steps, "tool_calls": tool_calls, "error": error, "detail": detail, "trace": messages}

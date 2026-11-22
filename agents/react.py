"""ReAct agent. Each turn the model emits ONE JSON object:
  {"thought": "...", "action": "<tool name>", "args": {...}}      -> run a tool
  {"thought": "...", "action": "finish", "message": "..."}         -> end the episode
Contract: run(instruction, db, tools, model, seed, fault_hook=None, max_steps=15, chat_fn=None) -> dict
returned dict: {"steps": int, "tool_calls": int, "error": str, "trace": list}
steps = number of LLM calls. error is "" | "max_steps" | "parse_error" | "api_error".
"""
import json

from contract import call_tool
from env.tools import POLICY_TEXT, tool_docs, tool_schemas

FORMAT_TEXT = """Each turn, call exactly ONE function (one of the tools above, or `finish`).
After each tool call you will receive an Observation. Call tools one at a time.
Call `finish` with your reply to the customer when the request is done, or when you must refuse.
(If you cannot call functions, reply with ONE JSON object instead:
 {"action": "<tool name>", "args": {...}}  or  {"action": "finish", "message": "..."} .)"""

MAX_PARSE_ERRORS = 3  # consecutive unparseable replies before giving up


def build_system_prompt() -> str:
    return f"{POLICY_TEXT}\n\nTools:\n{tool_docs()}\n\n{FORMAT_TEXT}"


def parse_action(text: str):
    """Return the first JSON object in text as a dict, or None."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "action" in obj:
                return obj
    return None


def run(instruction, db, tools, model, seed, fault_hook=None, max_steps=15, chat_fn=None):
    if chat_fn is None:
        import llm
        chat_fn = llm.chat

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": instruction},
    ]
    schemas = tool_schemas()
    steps = tool_calls = parse_fails = 0
    error = detail = ""

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
            break
        args = act.get("args", {})
        if not isinstance(args, dict):
            obs = "ERROR: args must be a JSON object"
        else:
            obs = call_tool(tools, str(name), db, args, fault_hook)
            tool_calls += 1
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    return {"steps": steps, "tool_calls": tool_calls, "error": error, "detail": detail, "trace": messages}

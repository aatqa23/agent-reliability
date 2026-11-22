"""Tests for tool-call -> JSON-action conversion in llm.py (no network). Run: python test_llm.py"""
import json
import os
from types import SimpleNamespace as NS

os.environ["LLM_CACHE"] = "0"      # these tests exercise the live path; test_cache.py covers the cache
import llm
from env.tools import tool_schemas


def expect(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    assert cond, label


def tc(name, args):
    return NS(function=NS(name=name, arguments=args if isinstance(args, str) else json.dumps(args)))


def msg(content="", calls=None):
    return NS(content=content, tool_calls=calls)


# conversion
out = json.loads(llm.message_to_text(msg("", [tc("get_order", {"order_id": "O1001"})])))
expect("tool call -> action", out == {"thought": "", "action": "get_order", "args": {"order_id": "O1001"}})

out = json.loads(llm.message_to_text(msg("done", [tc("finish", {"message": "Cancelled."})])))
expect("finish call -> finish action", out["action"] == "finish" and out["message"] == "Cancelled.")

out = json.loads(llm.message_to_text(msg("", [tc("get_order", "not json")])))
expect("bad arguments kept raw", out["args"] == "not json")

out = json.loads(llm.message_to_text(msg("", [tc("a", {}), tc("b", {})])))
expect("only first call used", out["action"] == "a")

expect("plain text passes through", llm.message_to_text(msg("hello")) == "hello")
expect("empty content -> empty string", llm.message_to_text(msg(None)) == "")

# schemas
names = [s["function"]["name"] for s in tool_schemas()]
expect("schemas include all tools + finish",
       names == ["lookup_customer", "get_order", "cancel_order", "modify_order_address", "refund_order", "finish"])

# chat() with a fake client: passes tools + tool_choice, returns converted text
seen = {}


class FakeCompletions:
    def create(self, **kw):
        seen.update(kw)
        return NS(choices=[NS(message=msg("", [tc("cancel_order", {"order_id": "O1001", "reason": "x"})]))])


llm._client = NS(chat=NS(completions=FakeCompletions()))
res = json.loads(llm.chat([{"role": "user", "content": "hi"}], "m", tools=tool_schemas()))
expect("chat converts tool call", res["action"] == "cancel_order")
expect("chat sends tools, does not force tool_choice", "tools" in seen and "tool_choice" not in seen)

seen.clear()
llm.chat([{"role": "user", "content": "hi"}], "m")
expect("chat without tools sends no tool_choice", "tools" not in seen and "tool_choice" not in seen)

# --- error handling -------------------------------------------------------
import openai

llm.time.sleep = lambda s: None            # no real waiting in tests


def fake_response(status):
    # minimal stand-in for an httpx response (the openai error classes only read these fields)
    return NS(request=NS(), status_code=status, headers={})


def bad_request(fg=None):
    return openai.BadRequestError("tool_use_failed", response=fake_response(400),
                                  body={"code": "tool_use_failed", "failed_generation": fg})


def rate_limit():
    return openai.RateLimitError("slow down", response=fake_response(429), body=None)


calls = {"n": 0}


class AlwaysBad:
    def create(self, **kw):
        calls["n"] += 1
        raise bad_request('{"action": "finish", "message": "x"}')


llm._client = NS(chat=NS(completions=AlwaysBad()))
out = llm.chat([{"role": "user", "content": "hi"}], "m", tools=tool_schemas())
expect("3 bad requests -> returns failed_generation text", out == '{"action": "finish", "message": "x"}')
expect("stops after exactly 3 attempts", calls["n"] == 3)


class FlakyThenOk:
    def __init__(self):
        self.n = 0

    def create(self, **kw):
        self.n += 1
        if self.n < 3:
            raise rate_limit()
        return NS(choices=[NS(message=msg("ok"))])


llm._client = NS(chat=NS(completions=FlakyThenOk()))
expect("rate limit retried until success", llm.chat([{"role": "user", "content": "hi"}], "m") == "ok")


class AlwaysDown:
    def create(self, **kw):
        raise rate_limit()


llm._client = NS(chat=NS(completions=AlwaysDown()))
try:
    llm.chat([{"role": "user", "content": "hi"}], "m")
    expect("persistent rate limit raises", False)
except RuntimeError:
    expect("persistent rate limit raises RuntimeError", True)

print("\nAll LLM tests passed.")

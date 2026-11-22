"""Tests for the disk cache in llm.py (fake API client, no network). Run: python test_cache.py"""
import json
import os
import shutil
import tempfile
import threading
from types import SimpleNamespace as NS

import openai

import llm


def _load(p):
    with open(p) as f:
        return json.load(f)


def expect(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    assert cond, label


tmp = tempfile.mkdtemp()
llm.CACHE_DIR = os.path.join(tmp, "cache")
os.environ.pop("LLM_CACHE", None)

calls = []
script = {"mode": "ok"}


def fake_create(**kw):
    calls.append(kw)
    if script["mode"] == "bad_request":
        e = openai.BadRequestError.__new__(openai.BadRequestError)
        Exception.__init__(e, "tool_use_failed")
        e.body = {"failed_generation": "garbled"}
        raise e
    if script["mode"] == "boom":
        raise RuntimeError("down")
    return NS(choices=[NS(message=NS(content=f"reply#{len(calls)}", tool_calls=None))])


llm._get = lambda: NS(chat=NS(completions=NS(create=fake_create)))
llm.time.sleep = lambda s: None          # no real waiting in retries

M = [{"role": "user", "content": "hi"}]
TL = [{"type": "function", "function": {"name": "f"}}]

r1 = llm.chat(M, "m1", 0.7, 0, TL)
expect("first call goes to the API", len(calls) == 1 and r1 == "reply#1")
r2 = llm.chat(M, "m1", 0.7, 0, TL)
expect("same key -> served from cache, no API call", len(calls) == 1 and r2 == r1)
expect("cache stats count hit and miss", llm.cache_stats() == {"hits": 1, "misses": 1})
expect("one file on disk", len([f for f in os.listdir(llm.CACHE_DIR) if f.endswith(".json")]) == 1)

llm.chat(M, "m1", 0.7, 1, TL)
expect("different seed -> miss (independent trials)", len(calls) == 2)
llm.chat(M, "m2", 0.7, 0, TL)
expect("different model -> miss", len(calls) == 3)
llm.chat(M, "m1", 0.2, 0, TL)
expect("different temperature -> miss", len(calls) == 4)
llm.chat(M, "m1", 0.7, 0, None)
expect("different tools -> miss", len(calls) == 5)
llm.chat(M + [{"role": "user", "content": "more"}], "m1", 0.7, 0, TL)
expect("different messages -> miss", len(calls) == 6)
llm.chat(M, "m1", 0.7, None, TL)
expect("seed=None is its own key", len(calls) == 7)
llm.chat(M, "m1", 0.7, 0, TL)
expect("original key still cached", len(calls) == 7)
expect("API call passes reasoning_effort=low and no seed",
       calls[0]["reasoning_effort"] == "low" and "seed" not in calls[0])

# survives a 'restart' (nothing in memory but the files)
n = len(calls)
llm._stats["hits"] = llm._stats["misses"] = 0
expect("hit after restart", llm.chat(M, "m1", 0.7, 0, TL) == "reply#1" and len(calls) == n)

# failures are not cached
script["mode"] = "bad_request"
before = len(os.listdir(llm.CACHE_DIR))
out = llm.chat([{"role": "user", "content": "x"}], "m1", 0.7, 0, TL)
expect("tool_use_failed fallback returned to agent", out == "garbled")
expect("fallback text NOT cached", len(os.listdir(llm.CACHE_DIR)) == before)
script["mode"] = "boom"
try:
    llm.chat([{"role": "user", "content": "y"}], "m1", 0.7, 0, TL)
    expect("exhausted retries raise", False)
except RuntimeError:
    expect("exhausted retries raise, nothing cached", len(os.listdir(llm.CACHE_DIR)) == before)
script["mode"] = "ok"
out = llm.chat([{"role": "user", "content": "x"}], "m1", 0.7, 0, TL)
expect("after failure the next call is live and then cached", out.startswith("reply#"))

# corrupt cache file is treated as a miss and repaired
key = llm.cache_key(M, "m1", 0.7, 0, TL)
with open(llm._cache_path(key), "w") as f:
    f.write("{not json")
n = len(calls)
llm.chat(M, "m1", 0.7, 0, TL)
expect("corrupt entry -> miss, rewritten", len(calls) == n + 1 and _load(llm._cache_path(key))["reply"])

# switch off
os.environ["LLM_CACHE"] = "0"
n = len(calls)
llm.chat(M, "m1", 0.7, 0, TL)
llm.chat(M, "m1", 0.7, 0, TL)
expect("LLM_CACHE=0 bypasses the cache", len(calls) == n + 2)
os.environ.pop("LLM_CACHE")

# parallel writers do not corrupt entries
errs = []
def worker(i):
    try:
        for j in range(30):
            llm.chat([{"role": "user", "content": f"p{j}"}], "m1", 0.7, 0, None)
    except Exception as e:
        errs.append(e)
ts = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
[t.start() for t in ts]; [t.join() for t in ts]
files = [f for f in os.listdir(llm.CACHE_DIR)]
expect("parallel use: no errors, no leftover tmp files, all entries valid",
       not errs and not any(f.endswith(".tmp") for f in files)
       and all(isinstance(_load(os.path.join(llm.CACHE_DIR, f))["reply"], str) for f in files))

# Windows behaviour: os.replace raises PermissionError while the target exists -> entry must still be valid, no .tmp left
real_replace = os.replace
def win_replace(src, dst):
    if os.path.exists(dst):
        raise PermissionError(5, "Access is denied")
    return real_replace(src, dst)
llm.chat([{"role": "user", "content": "w"}], "m1", 0.7, 0, None)
os.replace = win_replace
llm._cache_put(llm.cache_key([{"role": "user", "content": "w"}], "m1", 0.7, 0, None), "other", "m1", 0)
os.replace = real_replace
files = os.listdir(llm.CACHE_DIR)
expect("simulated Windows PermissionError: no error, no leftover .tmp, original entry kept",
       not any(f.endswith(".tmp") for f in files)
       and _load(llm._cache_path(llm.cache_key([{"role": "user", "content": "w"}], "m1", 0.7, 0, None)))["reply"] != "other")

shutil.rmtree(tmp)
print("\nAll cache tests passed.")

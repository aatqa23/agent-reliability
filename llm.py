"""LLM client. Contract: chat(messages, model, temperature=0.7, seed=None, tools=None) -> str.
If `tools` (OpenAI function schemas) is given and the model calls a function, the call is
converted to the agent's JSON action text (a plain-text reply is returned unchanged):
   {"thought": ..., "action": <name>, "args": {...}}   or   {"thought": ..., "action": "finish", "message": ...}

DISK CACHE (added by B). Every successful reply is stored in results/cache/<sha256>.json, keyed on
(messages, model, temperature, seed, tools, reasoning effort). Same key -> the stored reply is returned
without calling the API, so interrupted runs resume for free and re-runs are reproducible.
  * The seed is part of the key, so the 5 trials of a task (seeds 0-4) are still 5 independent samples.
    (The seed is NOT sent to the API; it only separates cache entries, exactly like the uncached version.)
  * Failures are never served from the cache in normal runs: exceptions, and the 'tool_use_failed' fallback text,
    are retried live next time. (A: the fallback text IS written, flagged "fallback": true, but only offline
    replays (LLM_OFFLINE=1) read it, so a replay reproduces the episode exactly. A later live success overwrites it.)
Environment switches:  LLM_CACHE=0 disables the cache,  LLM_CACHE_DIR=<dir> changes its location,
LLM_OFFLINE=1 never calls the API: a cache miss raises CacheMiss at once (offline replays, no key needed).

TOKEN ACCOUNTING (added by A). Every call records the API's usage (prompt/completion tokens) in a per-thread
counter; run_episode() calls usage_reset() before an episode and usage_get() after it. Cache entries store the
usage of the reply they hold, so a cached re-run still reports what the episode cost when it ran live
(`*_tokens`), while `live_tokens` is what was actually spent this time. Cache entries written before this change
have no usage; such hits are counted in `untracked_calls`. Tokens of rejected generations (Groq 400
tool_use_failed) are not reported by the API and so are not counted.
"""
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
MODELS = {"large": "openai/gpt-oss-120b", "small": "openai/gpt-oss-20b",
          "qwen": "qwen/qwen3.8-27b"}          # different family (A, phase 3); same temperature, tools, policies
_client = None


def _get():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                         base_url="https://api.groq.com/openai/v1")
    return _client


def message_to_text(msg) -> str:
    """Turn an API message into the agent's text format."""
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        tc = tcs[0]                      # one call per turn; extras are ignored
        name = tc.function.name
        raw = tc.function.arguments or "{}"
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = raw                   # agent reports "args must be a JSON object"
        thought = msg.content or ""
        if name == "finish":
            text = args.get("message", "") if isinstance(args, dict) else ""
            return json.dumps({"thought": thought, "action": "finish", "message": text})
        return json.dumps({"thought": thought, "action": name, "args": args})
    return msg.content or ""


# ---------------------------------------------------------------- token accounting (A)
_usage = threading.local()
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "live_tokens",
                "api_calls", "cache_hits", "untracked_calls")


def usage_reset() -> None:
    """Start counting for the current thread (one episode)."""
    _usage.d = dict.fromkeys(USAGE_FIELDS, 0)


def usage_get() -> dict:
    """Totals for the current thread since the last usage_reset()."""
    return dict(getattr(_usage, "d", None) or dict.fromkeys(USAGE_FIELDS, 0))


def _usage_add(usage: dict | None, live: bool) -> None:
    d = getattr(_usage, "d", None)
    if d is None:
        usage_reset()
        d = _usage.d
    d["api_calls" if live else "cache_hits"] += 1
    if not usage:
        if not live:
            d["untracked_calls"] += 1
        return
    p, c = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    t = int(usage.get("total_tokens") or (p + c))
    d["prompt_tokens"] += p
    d["completion_tokens"] += c
    d["total_tokens"] += t
    if live:
        d["live_tokens"] += t


def _usage_of(resp) -> dict | None:
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    return {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0}


def _chat_live(messages: list, model: str, temperature: float = 0.7,
               tools: list | None = None) -> tuple[str, bool, dict | None]:
    """One live API call with A's retry logic. Returns (text, cacheable, usage)."""
    last = None
    bad_requests = 0
    for attempt in range(6):
        try:
            kwargs = dict(model=model, messages=messages,
                          temperature=temperature, reasoning_effort="low")
            if tools:
                kwargs["tools"] = tools      # tool_choice left on auto: plain-text replies are allowed
            r = _get().chat.completions.create(**kwargs)
            return message_to_text(r.choices[0].message), True, _usage_of(r)
        except openai.BadRequestError as e:
            # Typically Groq's "tool_use_failed": the model produced a malformed tool call.
            # Retry immediately (sampling is random); after 3 failures hand back whatever the
            # model generated so the agent counts it as a parse failure and can recover.
            last = e
            bad_requests += 1
            print(f"[llm] bad request ({bad_requests}/3): {str(e)[:200]}", file=sys.stderr)
            if bad_requests >= 3:
                body = getattr(e, "body", None)
                fg = body.get("failed_generation") if isinstance(body, dict) else None
                return (fg or ""), False, None      # a failed generation: hand it back but do not cache it
        except Exception as e:          # rate limits, timeouts, 5xx
            last = e
            wait = min(2 ** attempt, 30)
            print(f"[llm] {type(e).__name__}: {str(e)[:200]} -- retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after retries: {last}")


# ---------------------------------------------------------------- disk cache (B)
REASONING_EFFORT = "low"
CACHE_DIR = os.environ.get("LLM_CACHE_DIR", os.path.join("results", "cache"))
_stats = {"hits": 0, "misses": 0}
_stats_lock = threading.Lock()


class CacheMiss(RuntimeError):
    """Raised in offline mode (LLM_OFFLINE=1) when a reply is not in the cache."""


def offline() -> bool:
    return os.environ.get("LLM_OFFLINE", "0") == "1"


def cache_enabled() -> bool:
    return os.environ.get("LLM_CACHE", "1") != "0"


def cache_key(messages: list, model: str, temperature: float, seed, tools) -> str:
    payload = {"messages": messages, "model": model, "temperature": temperature, "seed": seed,
               "tools": tools, "reasoning_effort": REASONING_EFFORT}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, key + ".json")


def _cache_get(key: str):
    """Returns (reply, usage) or None. usage is None for entries written before token accounting."""
    try:
        with open(_cache_path(key)) as f:
            d = json.load(f)
        if not isinstance(d.get("reply"), str):
            return None
        if d.get("fallback") and not offline():
            return None      # a failed generation: live runs retry it, only offline replays reuse it
        u = d.get("usage")
        return d["reply"], (u if isinstance(u, dict) else None)
    except (OSError, ValueError, KeyError):
        return None          # missing or corrupt entry -> treat as a miss


def _cache_put(key: str, reply: str, model: str, seed, usage: dict | None = None, fallback: bool = False) -> None:
    tmp = None
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            entry = {"reply": reply, "model": model, "seed": seed, "usage": usage}
            if fallback:
                entry["fallback"] = True
            json.dump(entry, f)
        target = _cache_path(key)
        for attempt in range(5):
            try:
                os.replace(tmp, target)             # atomic where the OS allows it
                tmp = None
                return
            except PermissionError:
                # Windows refuses to replace a file another thread is replacing/reading at that moment.
                if os.path.exists(target):
                    return                          # someone else just stored this key: nothing left to do
                time.sleep(0.05 * (attempt + 1))
    except OSError as e:
        print(f"[llm] cache write failed: {e}", file=sys.stderr)      # never let caching break a run
    finally:
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass


def cache_stats() -> dict:
    with _stats_lock:
        return dict(_stats)


def chat(messages: list, model: str, temperature: float = 0.7, seed: int | None = None,
         tools: list | None = None) -> str:
    if not cache_enabled():
        text, _, usage = _chat_live(messages, model, temperature, tools)
        _usage_add(usage, live=True)
        return text
    key = cache_key(messages, model, temperature, seed, tools)
    hit = _cache_get(key)
    if hit is not None:
        with _stats_lock:
            _stats["hits"] += 1
        _usage_add(hit[1], live=False)
        return hit[0]
    if offline():
        raise CacheMiss(f"LLM_OFFLINE=1 and no cached reply for key {key[:12]} in {CACHE_DIR}")
    text, cacheable, usage = _chat_live(messages, model, temperature, tools)
    with _stats_lock:
        _stats["misses"] += 1
    _usage_add(usage, live=True)
    _cache_put(key, text, model, seed, usage, fallback=not cacheable)
    return text

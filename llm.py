"""LLM client. Contract: chat(messages, model, temperature=0.7, seed=None, tools=None) -> str.
If `tools` (OpenAI function schemas) is given and the model calls a function, the call is
converted to the agent's JSON action text (a plain-text reply is returned unchanged):
   {"thought": ..., "action": <name>, "args": {...}}   or   {"thought": ..., "action": "finish", "message": ...}

DISK CACHE (added by B). Every successful reply is stored in results/cache/<sha256>.json, keyed on
(messages, model, temperature, seed, tools, reasoning effort). Same key -> the stored reply is returned
without calling the API, so interrupted runs resume for free and re-runs are reproducible.
  * The seed is part of the key, so the 5 trials of a task (seeds 0-4) are still 5 independent samples.
    (The seed is NOT sent to the API; it only separates cache entries, exactly like the uncached version.)
  * Failures are never cached: exceptions, and the 'tool_use_failed' fallback text, are retried live next time.
Environment switches:  LLM_CACHE=0 disables the cache,  LLM_CACHE_DIR=<dir> changes its location.
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
MODELS = {"large": "openai/gpt-oss-120b", "small": "openai/gpt-oss-20b"}
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


def _chat_live(messages: list, model: str, temperature: float = 0.7,
               tools: list | None = None) -> tuple[str, bool]:
    """One live API call with A's retry logic. Returns (text, cacheable)."""
    last = None
    bad_requests = 0
    for attempt in range(6):
        try:
            kwargs = dict(model=model, messages=messages,
                          temperature=temperature, reasoning_effort="low")
            if tools:
                kwargs["tools"] = tools      # tool_choice left on auto: plain-text replies are allowed
            r = _get().chat.completions.create(**kwargs)
            return message_to_text(r.choices[0].message), True
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
                return (fg or ""), False      # a failed generation: hand it back but do not cache it
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
    try:
        with open(_cache_path(key)) as f:
            d = json.load(f)
        return d["reply"] if isinstance(d.get("reply"), str) else None
    except (OSError, ValueError, KeyError):
        return None          # missing or corrupt entry -> treat as a miss


def _cache_put(key: str, reply: str, model: str, seed) -> None:
    tmp = None
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump({"reply": reply, "model": model, "seed": seed}, f)
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
        return _chat_live(messages, model, temperature, tools)[0]
    key = cache_key(messages, model, temperature, seed, tools)
    hit = _cache_get(key)
    if hit is not None:
        with _stats_lock:
            _stats["hits"] += 1
        return hit
    text, cacheable = _chat_live(messages, model, temperature, tools)
    with _stats_lock:
        _stats["misses"] += 1
    if cacheable:
        _cache_put(key, text, model, seed)
    return text

"""Fault injector (B). Contract used by run_episode.py:

    make_fault_hook(rate: float, rng: random.Random) -> hook(tool_name: str, args: dict) -> str | None

The hook is passed to contract.call_tool(..., fault_hook=hook). On every tool call it either returns None
(no fault: the tool runs normally) or an error string (fault: the tool is NOT executed, the string is
the observation the agent sees). So an injected fault is always a *transient, pre-execution* failure:
the database is left untouched and a sensible agent can simply retry the same call.

  rate  = probability that any single tool call fails (E3 uses 0.1 and 0.2; 0.0 never injects).
  fault kinds (chosen uniformly among the three when a fault fires):
     timeout    - the call timed out
     rate_limit - the backend answered HTTP 429
     malformed  - the backend answered with a truncated / garbled payload (not usable output)

Reproducibility: the hook consumes exactly THREE random numbers from `rng` per tool call (whether or not a
fault fires), so for a fixed rng seed the fault pattern depends only on how many tool calls the agent makes.
run_episode seeds rng from "<task_id>|<seed>|<rate>", so every episode has its own reproducible fault stream.
"""
import random
from typing import Callable, Optional

FAULT_KINDS = ("timeout", "rate_limit", "malformed")

_TIMEOUT = [
    "ERROR: request timed out after 30s (TimeoutError). The operation was not completed.",
    "ERROR: upstream service did not respond within 30s (timeout). No changes were made.",
]
_RATE_LIMIT = [
    "ERROR: 429 Too Many Requests - rate limit exceeded, retry after 2s.",
    "ERROR: rate limit reached for this endpoint (HTTP 429). Please retry shortly.",
]
_MALFORMED = [
    '{"status": "ok", "data": {"order_id": "O1',                       # truncated JSON
    '��{"result": null, "payload": <<corrupted>>',            # garbled bytes
    'ERROR: malformed response from backend (could not parse payload).',
]
_MESSAGES = {"timeout": _TIMEOUT, "rate_limit": _RATE_LIMIT, "malformed": _MALFORMED}


def make_fault_hook(rate: float, rng: random.Random) -> Callable[[str, dict], Optional[str]]:
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"fault rate must be in [0, 1], got {rate}")
    counts = {"calls": 0, "faults": 0, **{k: 0 for k in FAULT_KINDS}}

    def hook(tool_name: str, args: dict) -> Optional[str]:
        u = rng.random()                      # always draw all three numbers -> stable stream
        kind = FAULT_KINDS[int(rng.random() * len(FAULT_KINDS))]
        v = rng.random()
        counts["calls"] += 1
        if u >= rate:
            return None
        msgs = _MESSAGES[kind]
        counts["faults"] += 1
        counts[kind] += 1
        return msgs[int(v * len(msgs))]

    hook.counts = counts                      # inspectable: hook.counts["faults"] / hook.counts["calls"]
    return hook

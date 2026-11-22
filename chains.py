"""Chain composer (B): join tasks that touch DIFFERENT orders into one longer task (E4).

  python chains.py --build            write data/chains.json (frozen; deterministic for a fixed task set + seed)
  python chains.py --check            validate the frozen file (no API calls)

A chain of length L is one task dict in the normal contract format:
    id           "A01+B03+B07"                 (member ids joined with "+")
    owner        owner of the first member
    instruction  a short wrapper + the L member instructions, each verbatim, as "Message 1: ..." "Message 2: ..."
    assertions   the member assertions concatenated (success = ALL hold, so chain success ~ product of members)
    chain        the list of member ids   (extra key, ignored by the rest of the pipeline)
Rules enforced when building:
  * members touch disjoint sets of orders (orders named in the instruction or in an assertion path);
  * no member twice, no chain twice;
  * a wrong-customer refusal member (the speaker asks about another customer's order) is never placed in a chain
    with a member spoken by that order's owner, otherwise the chain itself would blur who is asking;
  * coverage is balanced: every task is used about equally often (least-used tasks are tried first).
The wrapper tells the agent the messages come from separate customers and that a refusal must NOT end the conversation
(the agent's normal policy text says "explain briefly and finish", which in a chain would silently skip every later
message); each message still states who it is from,
so policy 4 (verify the order belongs to the customer who is asking) stays well defined inside a chain.
"""
import argparse
import json
import os
import random
import re
import sys

from contract import load_base_db, validate_task
from taskset import load_all_tasks

CHAINS_PATH = os.path.join("data", "chains.json")
DEFAULT_LENS = (2, 3, 4)
DEFAULT_PER_LEN = 10
DEFAULT_SEED = 2026

WRAPPER = ("You will receive {n} separate customer messages in a row. Handle every message, one after the other, "
           "following the policies. If a message must be refused, explain the refusal in your final reply but KEEP GOING "
           "with the remaining messages: do not call finish until you have dealt with all {n} messages. "
           "Then finish with one reply that covers all of them.")


def orders_of(task: dict) -> set:
    return set(re.findall(r"O\d{4}", task["instruction"])) | {a["path"].split(".")[1] for a in task["assertions"]}


def speaker_of(task: dict):
    m = re.search(r"\bC\d\b", task["instruction"])
    return m.group(0) if m else None


def foreign_owners(task: dict, base: dict) -> set:
    """Owners of the orders this task mentions, other than the speaker (non-empty = wrong-customer request)."""
    me = speaker_of(task)
    return {base["orders"][o]["customer_id"] for o in orders_of(task) if o in base["orders"]} - {me}


def compatible(a: dict, b: dict, base: dict) -> bool:
    if orders_of(a) & orders_of(b):
        return False
    if speaker_of(b) in foreign_owners(a, base) or speaker_of(a) in foreign_owners(b, base):
        return False
    return True


def compose(members: list) -> dict:
    n = len(members)
    parts = [WRAPPER.format(n=n)] + [f"Message {i + 1}:\n{m['instruction']}" for i, m in enumerate(members)]
    chain = {
        "id": "+".join(m["id"] for m in members),
        "owner": members[0]["owner"],
        "instruction": "\n\n".join(parts),
        "assertions": [a for m in members for a in m["assertions"]],
        "paraphrases": [],
        "chain": [m["id"] for m in members],
    }
    if all(m.get("refusal") for m in members):
        chain["refusal"] = True
    return chain


def build_chains(tasks: list, base: dict, lens=DEFAULT_LENS, per_len: int = DEFAULT_PER_LEN,
                 seed: int = DEFAULT_SEED) -> list:
    rng = random.Random(seed)
    usage = {t["id"]: 0 for t in tasks}
    seen, out = set(), []
    for L in lens:
        made, attempts = 0, 0
        while made < per_len:
            attempts += 1
            if attempts > 5000:
                raise RuntimeError(f"could not build {per_len} distinct chains of length {L} from {len(tasks)} tasks")
            order = sorted(tasks, key=lambda t: (usage[t["id"]], rng.random()))      # least used first, random ties
            members = []
            for t in order:
                if all(compatible(t, m, base) for m in members):
                    members.append(t)
                    if len(members) == L:
                        break
            if len(members) < L or frozenset(m["id"] for m in members) in seen:
                # perturb: a random start also gives variety once the greedy choice repeats
                rng.shuffle(order)
                members = []
                for t in order:
                    if all(compatible(t, m, base) for m in members):
                        members.append(t)
                        if len(members) == L:
                            break
                if len(members) < L or frozenset(m["id"] for m in members) in seen:
                    continue
            seen.add(frozenset(m["id"] for m in members))
            for m in members:
                usage[m["id"]] += 1
            rng.shuffle(members)
            out.append(compose(members))
            made += 1
    return out


def save_chains(chains: list, tasks: list, seed: int, path: str = CHAINS_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    doc = {"meta": {"seed": seed, "source_tasks": sorted(t["id"] for t in tasks),
                    "lens": sorted({len(c["chain"]) for c in chains})},
           "chains": chains}
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_chains(path: str = CHAINS_PATH, lens=None) -> list:
    with open(path) as f:
        chains = json.load(f)["chains"]
    return [c for c in chains if lens is None or len(c["chain"]) in lens]


def stale_reason(path: str = CHAINS_PATH, tasks: list | None = None):
    """None if the frozen chains were built from the tasks now on disk, else a human-readable reason."""
    if not os.path.exists(path):
        return f"{path} does not exist (run: python chains.py --build)"
    tasks = load_all_tasks(with_paraphrases=False) if tasks is None else tasks
    with open(path) as f:
        meta = json.load(f)["meta"]
    now = sorted(t["id"] for t in tasks)
    if meta["source_tasks"] != now:
        new = sorted(set(now) - set(meta["source_tasks"]))
        gone = sorted(set(meta["source_tasks"]) - set(now))
        return f"chains were built from a different task set (new: {new}, removed: {gone})"
    return None


def check_chains(chains: list, tasks: list, base: dict) -> list:
    by_id = {t["id"]: t for t in tasks}
    problems, ids = [], set()
    for c in chains:
        errs = validate_task(c, base)
        if c["id"] in ids:
            errs.append("duplicate chain")
        ids.add(c["id"])
        mem = [by_id.get(i) for i in c["chain"]]
        if None in mem:
            errs.append("unknown member task")
        else:
            if len(set(c["chain"])) != len(c["chain"]):
                errs.append("repeated member")
            for i in range(len(mem)):
                for j in range(i + 1, len(mem)):
                    if not compatible(mem[i], mem[j], base):
                        errs.append(f"members {mem[i]['id']} and {mem[j]['id']} are not compatible (shared order or identity clash)")
            if c["assertions"] != [a for m in mem for a in m["assertions"]]:
                errs.append("assertions are not the concatenation of the members' assertions")
            if any(m["instruction"] not in c["instruction"] for m in mem):
                errs.append("a member instruction is missing from the chain instruction")
        problems += [f"{c['id']}: {e}" for e in errs]
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lens", type=int, nargs="+", default=list(DEFAULT_LENS))
    ap.add_argument("--per-len", type=int, default=DEFAULT_PER_LEN)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=CHAINS_PATH)
    a = ap.parse_args(argv)
    tasks, base = load_all_tasks(with_paraphrases=False), load_base_db()

    if a.build:
        chains = build_chains(tasks, base, a.lens, a.per_len, a.seed)
        save_chains(chains, tasks, a.seed, a.out)
        used = {}
        for c in chains:
            for i in c["chain"]:
                used[i] = used.get(i, 0) + 1
        print(f"Wrote {len(chains)} chains to {a.out} from {len(tasks)} tasks "
              f"({', '.join(f'{sum(len(c['chain']) == L for c in chains)} of length {L}' for L in a.lens)}).")
        print("Task usage (min/max):", min(used.values()), "/", max(used.values()), f"| unused tasks: {sorted(set(t['id'] for t in tasks) - set(used))}")
    problems = check_chains(load_chains(a.out), tasks, base)
    stale = stale_reason(a.out, tasks)
    for p in problems:
        print("PROBLEM", p)
    if stale:
        print("WARNING:", stale)
    print(f"{'OK' if not problems else 'FAILED'}: {len(load_chains(a.out))} chains checked")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

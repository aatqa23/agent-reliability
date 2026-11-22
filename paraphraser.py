"""Paraphraser (B): 3 frozen rewordings per task -> data/paraphrases.json  {task_id: [3 strings]}.

  python paraphraser.py            fill in every task that has no paraphrases yet (uses the LLM, needs GROQ_API_KEY)
  python paraphraser.py --check    validate data/paraphrases.json against the current tasks (no API calls)
  python paraphraser.py --redo C03 regenerate one task
  python paraphraser.py --show     print originals next to their paraphrases for a manual read-through

A paraphrase is only accepted if validate_paraphrase() passes: it must keep every order id, customer id, amount
and address EXACTLY, add no new ids/amounts, differ from the original and from its siblings. Nothing else may
change (the meaning, including which requests are policy violations, must stay the same), so after generating,
read them once with --show. Because llm.chat is disk-cached and the seed is fixed, re-running reproduces the
same text: the paraphrases are frozen once written.
The paraphrases for A01-A10 and B01-B10 were written by hand and pass the same validator.
"""
import argparse
import json
import os
import re
import sys

from taskset import PARAPHRASES_PATH, load_all_tasks, load_paraphrases

N_PARAPHRASES = 3
MODEL = "openai/gpt-oss-120b"
STYLES = ["polite and formal", "short, casual and to the point", "re-ordered, giving the reason first and the request second"]

ID_RE = re.compile(r"\b[OC]\d{1,4}\b")
# an address as it appears in these tasks: "15 Clifton Block 4, Karachi", "45 Gulberg III, Lahore", ...
ADDR_RE = re.compile(r"\b\d+(?: (?:[A-Z][A-Za-z]*|\d+))+, [A-Z][a-z]+")
NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)")


def entities(text: str) -> dict:
    """Things a paraphrase must keep exactly: ids, addresses, amounts."""
    addrs = ADDR_RE.findall(text)
    rest = text
    for a in addrs:
        rest = rest.replace(a, " ")
    ids = set(ID_RE.findall(rest))
    nums = set(NUM_RE.findall(ID_RE.sub(" ", rest)))
    return {"ids": ids, "addresses": set(addrs), "numbers": nums}


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def validate_paraphrase(original: str, para: str) -> list:
    errs = []
    if not isinstance(para, str) or not para.strip():
        return ["empty"]
    o, p = entities(original), entities(para)
    for k in ("ids", "addresses", "numbers"):
        if o[k] - p[k]:
            errs.append(f"missing {k}: {sorted(o[k] - p[k])}")
        if p[k] - o[k]:
            errs.append(f"new {k}: {sorted(p[k] - o[k])}")
    if _norm(para) == _norm(original):
        errs.append("identical to the original")
    if len(para) > 3 * len(original) + 40:
        errs.append("much longer than the original")
    return errs


def validate_set(task: dict, paras: list) -> list:
    errs = []
    if len(paras) != N_PARAPHRASES:
        errs.append(f"expected {N_PARAPHRASES} paraphrases, got {len(paras)}")
    for i, p in enumerate(paras):
        errs += [f"#{i}: {e}" for e in validate_paraphrase(task["instruction"], p)]
    if len({_norm(p) for p in paras}) != len(paras):
        errs.append("duplicate paraphrases")
    return errs


def build_prompt(task: dict) -> list:
    styles = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(STYLES))
    system = ("You rewrite customer-service messages. You keep the exact meaning and change only the wording. "
              "You never add, drop or soften a request, and you never change who is asking.")
    user = (f"Rewrite the customer message below in {N_PARAPHRASES} different ways, one per style:\n{styles}\n\n"
            "Rules:\n"
            "- Keep every order id (like O1001), customer id (like C1), amount and address EXACTLY as written.\n"
            "- Keep every request, in the same sense, even if it looks unreasonable. Do not add facts or numbers.\n"
            "- Do not use digits for anything except the ids, amounts and addresses already in the message "
            "(write 'first', 'second' instead of 1), 2)).\n"
            "- Keep the customer's name and customer id.\n"
            f"Answer with ONLY a JSON list of {N_PARAPHRASES} strings.\n\nMessage:\n{task['instruction']}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_list(text: str):
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "[":
            try:
                obj, _ = dec.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return obj
    return None


def generate(task: dict, chat_fn=None, tries: int = 6) -> list:
    """Ask the LLM until 3 distinct valid paraphrases are collected (valid ones are kept across tries)."""
    if chat_fn is None:
        import llm
        chat_fn = llm.chat
    good = []
    for attempt in range(tries):
        reply = chat_fn(build_prompt(task), MODEL, temperature=0.7, seed=attempt, tools=None)
        for p in parse_list(reply) or []:
            p = p.strip()
            if not validate_paraphrase(task["instruction"], p) and _norm(p) not in {_norm(g) for g in good}:
                good.append(p)
        if len(good) >= N_PARAPHRASES:
            return good[:N_PARAPHRASES]
    raise RuntimeError(f"{task['id']}: only {len(good)} valid paraphrases after {tries} tries")


def save(paraphrases: dict, path: str = PARAPHRASES_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(sorted(paraphrases.items())), f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def check_all(tasks: list, paraphrases: dict) -> list:
    problems = []
    for t in tasks:
        if t["id"] not in paraphrases:
            problems.append(f"{t['id']}: no paraphrases")
        else:
            problems += [f"{t['id']}: {e}" for e in validate_set(t, paraphrases[t["id"]])]
    for tid in paraphrases:
        if tid not in {t["id"] for t in tasks}:
            problems.append(f"{tid}: paraphrases for an unknown task")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--redo", nargs="*", default=[], help="task ids to regenerate")
    a = ap.parse_args(argv)

    tasks = load_all_tasks(with_paraphrases=False)
    paras = load_paraphrases()

    if a.show:
        for t in tasks:
            print(f"\n{t['id']}  ORIGINAL: {t['instruction']}")
            for i, p in enumerate(paras.get(t["id"], [])):
                print(f"      P{i}: {p}")
        return 0
    if a.check:
        problems = check_all(tasks, paras)
        for p in problems:
            print("PROBLEM", p)
        print(f"{'OK' if not problems else 'FAILED'}: {len(tasks)} tasks, {sum(len(v) for v in paras.values())} paraphrases")
        return 1 if problems else 0

    todo = [t for t in tasks if t["id"] in a.redo or t["id"] not in paras]
    if not todo:
        print("Nothing to do: every task already has paraphrases.")
        return 0
    for t in todo:
        paras[t["id"]] = generate(t)
        save(paras)                         # write after every task: safe to interrupt
        print(f"{t['id']}: done")
    problems = check_all(tasks, paras)
    for p in problems:
        print("PROBLEM", p)
    print("Now read them once:  python paraphraser.py --show")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

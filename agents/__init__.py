"""Agent registry. Every agent exposes run(instruction, db, tools, model, seed,
fault_hook=None, max_steps=15, chat_fn=None) -> {"steps","tool_calls","error","trace"}.
C registers the second design here, e.g.  AGENTS["reflexion"] = reflexion.run
"""
from agents import plan_execute, react

AGENTS = {
    "react": react.run,
    "plan_execute": plan_execute.run,     # C: plan, execute, then a ledger-based completion audit
}

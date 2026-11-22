"""Print the RAW API message for one tool-calling request, to see where the model puts its output.
Usage: python debug_llm.py [model]"""
import sys

import llm
from env.tools import POLICY_TEXT, tool_schemas

model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-120b"
r = llm._get().chat.completions.create(
    model=model,
    messages=[{"role": "system", "content": POLICY_TEXT},
              {"role": "user", "content": "Hi, I'm Alice Khan (customer C1). Please cancel my order O1001."}],
    tools=tool_schemas(),
    temperature=0.7, reasoning_effort="low")
m = r.choices[0].message
print("content   :", repr(m.content))
print("tool_calls:", m.tool_calls)
print("reasoning :", repr(getattr(m, "reasoning", None))[:300])
print("finish    :", r.choices[0].finish_reason)

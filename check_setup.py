import os, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1")

MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

for m in MODELS:
    t = time.time()
    r = client.chat.completions.create(
        model=m,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        temperature=0.7,
        reasoning_effort="low",
    )
    print(f"{m} -> {r.choices[0].message.content!r} "
          f"({time.time()-t:.1f}s, {r.usage.total_tokens} tokens)")
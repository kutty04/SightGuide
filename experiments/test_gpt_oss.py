import os
from pathlib import Path
from groq import Groq

ROOT = Path(__file__).resolve().parent.parent
env_file = ROOT / ".env"
for line in env_file.read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.strip().split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

import sys
sys.path.insert(0, str(ROOT))
from experiments.test_prompt_comparison import REFINED_PROMPT_2

ev = """DIRECT VISUAL OBSERVATIONS FROM IMAGE PIXELS:
- the woman has black hair and is wearing a blue shirt
- the woman with long black hair and a blue shirt
- the woman's hair color is black, and she has dark brown hair

VERIFIED SCENE CONTEXT:
a portrait of a woman in a blue shirt

VERIFIED VISIBLE OCR TEXT:
[No visible text detected]"""

r = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": REFINED_PROMPT_2},
        {"role": "user", "content": f"EVIDENCE:\n{ev}\n\nUSER QUESTION: What color is her hair?"}
    ],
    max_tokens=600,
    temperature=0.1
)
print("Content:", repr(r.choices[0].message.content))
print("Finish reason:", r.choices[0].finish_reason)

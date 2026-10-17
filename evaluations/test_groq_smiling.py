import os
from pathlib import Path
from groq import Groq
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.app.services.qa_service import QA_SYSTEM_PROMPT_VISION, QA_MODEL, QA_TEMPERATURE, QA_MAX_TOKENS

env_path = Path('.env')
for line in env_path.read_text(encoding='utf-8').splitlines():
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

client = Groq(api_key=os.environ['GROQ_API_KEY'])

evidence = """DIRECT VISUAL OBSERVATIONS FROM IMAGE PIXELS:
- the girl is smiling and looking at the camera
- in this image, person looks like a girl with black hair and brown eyes she is wearing a blue shirt and a pair of earrings
- a photo of a woman with black hair and a blue shirt

VERIFIED SCENE CONTEXT:
a close-up portrait of a woman in a blue shirt

VERIFIED VISIBLE OCR TEXT:
[No visible text detected]"""

questions = [
    "Is the girl smiling?",
    "Is the person smiling?",
    "What facial expression is visible?",
    "Does the person appear to be smiling or not?"
]

for q in questions:
    user_content = f"EVIDENCE:\n{evidence}\n\nUSER QUESTION: {q}"
    resp = client.chat.completions.create(
        model=QA_MODEL,
        messages=[
            {"role": "system", "content": QA_SYSTEM_PROMPT_VISION},
            {"role": "user", "content": user_content}
        ],
        max_tokens=QA_MAX_TOKENS,
        temperature=QA_TEMPERATURE
    )
    print(f"Q: {q}")
    print(f"A: {resp.choices[0].message.content.strip()}\n")

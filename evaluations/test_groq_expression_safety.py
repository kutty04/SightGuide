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

scenarios = [
    (
        "NEUTRAL / SERIOUS",
        """DIRECT VISUAL OBSERVATIONS FROM IMAGE PIXELS:
- the woman has a serious or neutral facial expression
- the face is not smiling, with a calm and direct gaze
- a photo of a woman wearing a hoodie

VERIFIED SCENE CONTEXT:
a portrait of a woman in a hoodie"""
    ),
    (
        "OBSCURED / TURNED AWAY",
        """DIRECT VISUAL OBSERVATIONS FROM IMAGE PIXELS:
- the person is facing away from the camera, back of head shown
- the face and facial expression are not visible or occluded
- a photo of a person looking at a landscape

VERIFIED SCENE CONTEXT:
a person standing on a mountain looking at the sunset"""
    ),
    (
        "NO PERSON / LANDSCAPE",
        """DIRECT VISUAL OBSERVATIONS FROM IMAGE PIXELS:
- a wide landscape with mountains and clouds
- no person or face is present in the frame

VERIFIED SCENE CONTEXT:
mountains covered in clouds at dusk"""
    )
]

for title, evidence in scenarios:
    print(f"\n==================== SCENARIO: {title} ====================")
    user_content = f"EVIDENCE:\n{evidence}\n\nUSER QUESTION: Is the person smiling?"
    resp = client.chat.completions.create(
        model=QA_MODEL,
        messages=[
            {"role": "system", "content": QA_SYSTEM_PROMPT_VISION},
            {"role": "user", "content": user_content}
        ],
        max_tokens=QA_MAX_TOKENS,
        temperature=QA_TEMPERATURE
    )
    print("A:", resp.choices[0].message.content.strip())

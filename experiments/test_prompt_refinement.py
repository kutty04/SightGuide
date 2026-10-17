import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from groq import Groq

ROOT = Path(__file__).resolve().parent.parent
env_file = ROOT / ".env"
for line in env_file.read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.strip().split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

QA_SYSTEM_PROMPT_VISION = """You are SightGuide's accurate, descriptive accessibility visual assistant for blind and low-vision users.
You answer user questions using the DIRECT VISUAL OBSERVATIONS extracted from the image pixels by the computer vision model, supplemented by verified scene context and visible OCR text.

FIRST-RESPONSE DESCRIPTIVENESS & GROUNDING POLICY:
1. INFORMATIVE FIRST RESPONSE (NO ELABORATION FOLLOW-UP REQUIRED):
   - Answer the exact question directly first in the opening sentence.
   - When supported by the direct visual observations or scene context, accompany the direct answer with 1 or 2 relevant, clearly visible companion details about the SAME queried subject in 1 to 2 natural, spoken-friendly sentences:
     * Clothing queries: State the color and integrate visible garment type, style, pattern, or accompanying accessories (e.g., "She is wearing a blue shirt paired with visible earrings." or "She is wearing a white dress with a subtle floral pattern.").
     * Hair queries: State the hair color along with visible length, style, or texture (e.g., "Her hair is black and worn long.").
     * Facial expression queries: State the visible expression directly with appropriate caution (e.g., "Yes, she appears to be smiling." or "She has a neutral, serious expression and does not appear to be smiling."). Never speculate on internal psychological emotions or mood.
     * Object queries: State what the object is along with its visible color, distinguishing feature, material, or spatial placement (e.g., "There is a blue water bottle resting on the table.").
     * Surroundings/spatial queries: Describe the relevant spatial relationship or visible background setting directly.
2. STAY ON THE QUERIED SUBJECT:
   - For clothing queries: only describe clothing, color, pattern, or accompanying accessories. Do not talk about hair, face, or background.
   - For hair queries: only describe hair color, length, or style. Do not talk about clothing.
   - For facial expression queries: only describe visible facial cues or smile. Do not talk about clothing or hair.
   - For object queries: only describe the object and its immediate position.
3. STRICT ANTI-HALLUCINATION & HONESTY:
   - ONLY include details directly corroborated by the visual observations or verified scene context.
   - NEVER invent fabrics, textures, patterns, facial expressions, emotions, or invisible background items that are not in the evidence.
   - If only part of an item is visible, state what is clearly observable and note what is obscured.
4. OUT OF FRAME, OCCLUDED, OR ABSENT:
   - If the requested detail (such as shoes or lower body in a portrait/headshot, or an absent object) is outside the camera frame or absent from the evidence, state honestly:
     "Based on the visual evidence, [detail] cannot be seen / is out of frame in this image."
   - In headshots, portraits, or bust crops, never extrapolate or assume lower-body clothing.
5. EXPERIMENTAL DETECTOR EVIDENCE:
   - If EXPERIMENTAL OBJECT DETECTOR EVIDENCE is present, treat it as auxiliary, tentative support for object presence or general location.
   - Never treat lack of a detection as proof of absence."""

test_cases = [
    {
        "q": "What color is her shirt?",
        "evidence": """DIRECT VISUAL OBSERVATIONS FROM IMAGE PIXELS:
- the woman has black hair and is wearing a blue shirt
- the person is wearing a blue shirt
- in this image, person looks like a girl with black hair and brown eyes she is wearing a blue shirt and a pair of earring

VERIFIED SCENE CONTEXT:
a portrait of a woman in a blue shirt

VERIFIED VISIBLE OCR TEXT:
[No visible text detected]"""
    },
    {
        "q": "What is her dress color?",
        "evidence": """DIRECT VISUAL OBSERVATIONS FROM IMAGE PIXELS:
- she is wearing a white dress with floral pattern
- the dress is white with small flowers
- a woman in a summer dress standing in a garden

VERIFIED SCENE CONTEXT:
a woman in a floral summer dress

VERIFIED VISIBLE OCR TEXT:
[No visible text detected]"""
    },
    {
        "q": "What color is her hair?",
        "evidence": """DIRECT VISUAL OBSERVATIONS FROM IMAGE PIXELS:
- the woman has black hair and is wearing a blue shirt
- the woman with long black hair and a blue shirt
- the woman's hair color is black, and she has dark brown hair

VERIFIED SCENE CONTEXT:
a portrait of a woman in a blue shirt

VERIFIED VISIBLE OCR TEXT:
[No visible text detected]"""
    },
    {
        "q": "Is she smiling?",
        "evidence": """DIRECT VISUAL OBSERVATIONS FROM IMAGE PIXELS:
- the person is looking at the camera with a smile on her face
- in this image, person looks like a girl with black hair and brown eyes she is wearing a blue shirt and a pair of earring

VERIFIED SCENE CONTEXT:
a portrait of a woman in a blue shirt

VERIFIED VISIBLE OCR TEXT:
[No visible text detected]"""
    },
    {
        "q": "What shoes is she wearing?",
        "evidence": """DIRECT VISUAL OBSERVATIONS FROM IMAGE PIXELS:
- the woman has black hair and is wearing a blue shirt
- a close up portrait of a woman in a blue shirt

VERIFIED SCENE CONTEXT:
a portrait of a woman in a blue shirt

VERIFIED VISIBLE OCR TEXT:
[No visible text detected]"""
    }
]

for tc in test_cases:
    q = tc["q"]
    ev = tc["evidence"]
    r = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": QA_SYSTEM_PROMPT_VISION},
            {"role": "user", "content": f"EVIDENCE:\n{ev}\n\nUSER QUESTION: {q}"}
        ],
        max_tokens=600,
        temperature=0.1
    )
    ans = r.choices[0].message.content.strip().strip('"')
    print(f"\nQ: {q}\n  -> {ans}")

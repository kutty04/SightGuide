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

PROMPT_V3 = """You are SightGuide's accurate, descriptive accessibility visual assistant for blind and low-vision users.
You answer user questions using the DIRECT VISUAL OBSERVATIONS extracted from the image pixels by the computer vision model, supplemented by verified scene context and visible OCR text.

FIRST-RESPONSE DESCRIPTIVENESS & GROUNDING POLICY:
1. INFORMATIVE FIRST RESPONSE (NO ELABORATION FOLLOW-UP REQUIRED):
   - Answer the exact question directly first in the opening sentence.
   - When supported by the direct visual observations or scene context, accompany the direct answer with 1 or 2 relevant, clearly visible companion details strictly about the SAME queried attribute in 1 to 2 natural, spoken-friendly sentences:
     * Clothing queries: State the color and integrate visible garment type, style, pattern, or accompanying accessories (e.g., "She is wearing a blue shirt, paired with visible earrings." or "She is wearing a white dress with a subtle floral pattern.").
     * Hair queries: State the hair color along with visible length, style, or texture (e.g., "Her hair is black and worn long.").
     * Facial expression queries: State the visible expression directly with appropriate caution along with visible facial or gaze cues (e.g., "Yes, she appears to be smiling while looking at the camera." or "She has a neutral, serious expression with no visible smile."). Never speculate on internal psychological emotions, personality, or mood.
     * Object queries: State what the object is along with its visible color, distinguishing feature, material, or spatial placement (e.g., "There is a blue water bottle resting on the table.").
     * Surroundings/spatial queries: Describe the relevant spatial relationship or visible background setting directly.
2. STRICT SUBJECT-ATTRIBUTE FOCUS (NO UNRELATED ATTRIBUTE BLEED):
   - For facial expression queries: ONLY describe facial expression, smile, or gaze. DO NOT mention clothing, hair, or accessories.
   - For clothing queries: ONLY describe clothing, color, pattern, or accompanying accessories. DO NOT mention hair or face.
   - For hair queries: ONLY describe hair color, length, or style. DO NOT mention clothing or face.
   - For object queries: ONLY describe the object and its immediate position.
   - Synthesize the answer fluidly: avoid repeating the same statement twice. Keep answers focused (1 to 2 sentences).
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

ev = """DIRECT VISUAL OBSERVATIONS FROM IMAGE PIXELS:
- the person is looking at the camera with a smile on her face
- in this image, person looks like a girl with black hair and brown eyes she is wearing a blue shirt and a pair of earring

VERIFIED SCENE CONTEXT:
a portrait of a woman in a blue shirt

VERIFIED VISIBLE OCR TEXT:
[No visible text detected]"""

r = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": PROMPT_V3},
        {"role": "user", "content": f"EVIDENCE:\n{ev}\n\nUSER QUESTION: Is she smiling?"}
    ],
    max_tokens=600,
    temperature=0.1
)
print("Answer to 'Is she smiling?':")
print(r.choices[0].message.content.strip().strip('"'))

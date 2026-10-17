import os
import base64
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

env_path = Path('.env')
if env_path.exists():
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

from backend.app.services.qa_service import QAService
from backend.app.schemas import QARequest

qa_svc = QAService.get_instance()

# 1. Neutral expression: toa-heftiba
neutral_img = Image.open('dataset/toa-heftiba-O3ymvT7Wf9U-unsplash.jpg')
buf1 = io = Path('dataset/toa-heftiba-O3ymvT7Wf9U-unsplash.jpg').read_bytes()
data_neutral = f"data:image/jpeg;base64,{base64.b64encode(buf1).decode('utf-8')}"

# 2. Obscured face: portrait crop lower body strictly below chin
portrait = Image.open('evaluations/test_portrait_woman.png')
w, h = portrait.size
crop_lower = portrait.crop((0, 350, w, h))
import io
buf2 = io.BytesIO()
crop_lower.save(buf2, format='PNG')
data_obscured = f"data:image/png;base64,{base64.b64encode(buf2.getvalue()).decode('utf-8')}"

# 3. No person: clouds
buf3 = Path('dataset/sample-clouds-400x300.jpg').read_bytes()
data_clouds = f"data:image/jpeg;base64,{base64.b64encode(buf3).decode('utf-8')}"

cases = [
    ("Neutral woman", data_neutral, "a portrait of a woman wearing a hoodie"),
    ("Obscured face (torso crop)", data_obscured, "a close-up view of a person's blue shirt, torso only"),
    ("No person (clouds)", data_clouds, "clouds in the sky")
]

for title, uri, cap in cases:
    print(f"\n==================== TEST: {title} ====================")
    req = QARequest(
        question="Is the person smiling?",
        image_data=uri,
        scene_caption=cap,
        ocr_text=""
    )
    res = qa_svc.answer_question(req)
    print("Answer:", res.answer)
    print("Capability:", res.capability)
    print("Is Uncertain:", res.is_uncertain)
    print("Evidence used:", res.evidence_used)

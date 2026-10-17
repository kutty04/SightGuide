import os
import sys
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

# Load .env
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

from app.services.qa_service import QAService
from app.schemas import QARequest

qa = QAService.get_instance()
img_path = ROOT / "evaluations" / "test_portrait_woman.png"
data_uri = "data:image/png;base64," + base64.b64encode(img_path.read_bytes()).decode("ascii")

questions = [
    "What color is her shirt?",
    "Tell me elaborately about her shirt color",
    "What is she wearing?",
    "What color is her hair?",
    "Is she smiling?",
]

for q in questions:
    res = qa.answer_question(
        QARequest(
            question=q,
            image_data=data_uri,
            scene_caption="a portrait of a woman in a blue shirt",
            ocr_text=""
        )
    )
    print(f"QUESTION: {q}")
    print(f"MODEL: {res.model_name}")
    print(f"ANSWER: {res.answer}")
    print(f"EVIDENCE: {res.evidence_used}")
    print("-" * 60)

import os
import base64
from pathlib import Path
from PIL import Image

env_path = Path('.env')
if env_path.exists():
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.qa_service import QAService
from backend.app.schemas import QARequest

img_path = Path('evaluations/test_portrait_woman.png')
with open(img_path, 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')
data_uri = f'data:image/png;base64,{b64}'

qa_svc = QAService.get_instance()
questions = [
    "Is the girl smiling?",
    "Is the person smiling?",
    "What facial expression is visible?",
    "Does the person appear to be smiling or not?"
]

for q in questions:
    req = QARequest(
        question=q,
        image_data=data_uri,
        scene_caption='a close-up portrait of a woman in a blue shirt',
        ocr_text=''
    )
    res = qa_svc.answer_question(req)
    print(f"\n================ Question: {q} ================")
    print("Answer:", res.answer)
    print("Capability:", res.capability)
    print("Is Uncertain:", res.is_uncertain)
    print("Evidence used:", res.evidence_used)

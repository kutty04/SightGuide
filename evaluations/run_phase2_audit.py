import requests
import time
import json
from pathlib import Path

test_cases = [
    {
        'category': '1. Cropped portrait',
        'path': 'evaluations/test_portrait_woman.png',
        'question': 'What is the person wearing?'
    },
    {
        'category': '2. Full-body person',
        'path': 'dataset/sample-city-park-400x300.webp',
        'question': 'What are the people doing in the scene?'
    },
    {
        'category': '3. Document/Sign with readable text',
        'path': 'evaluations/eval_v3_grounding/images/08_storefront_visible_text.png',
        'question': 'What visible words or signs are shown?'
    },
    {
        'category': '4. Busy scene containing multiple objects',
        'path': 'evaluations/eval_v3_grounding/images/06_indoor_cluttered_desk.png',
        'question': 'What items are on the desk?'
    },
    {
        'category': '5. Image with no readable text',
        'path': 'dataset/sample-clouds-400x300.jpg',
        'question': 'Is there any readable text or signage in this picture?'
    },
    {
        'category': '6. Ambiguous or low-quality image',
        'path': 'evaluations/eval_v3_grounding/images/07_ambiguous_illustration.png',
        'question': 'Can you determine who this character is?'
    }
]

results = []
for tc in test_cases:
    p = Path(tc['path'])
    if not p.exists():
        print(f"MISSING FILE: {tc['path']}")
        continue
    
    t0 = time.time()
    # 1. Caption endpoint
    with open(p, 'rb') as f:
        r_cap = requests.post(
            'http://127.0.0.1:8000/api/caption/generate',
            files={'file': (p.name, f, 'image/png')},
            data={'lambda_weight': '0.5', 'method': 'proposed'}
        )
    cap_data = r_cap.json() if r_cap.status_code == 200 else {}
    
    # 2. OCR endpoint
    with open(p, 'rb') as f:
        r_ocr = requests.post(
            'http://127.0.0.1:8000/api/ocr/extract',
            files={'file': (p.name, f, 'image/png')}
        )
    ocr_data = r_ocr.json() if r_ocr.status_code == 200 else {}
    
    # 3. QA endpoint
    fused_text = cap_data.get('fusion', {}).get('fused_caption') or cap_data.get('selected_pair', {}).get('rank_1_caption')
    ocr_text = ocr_data.get('text') if ocr_data.get('has_text') else ''
    qa_req = {
        'question': tc['question'],
        'scene_caption': fused_text,
        'ocr_text': ocr_text,
        'history': []
    }
    r_qa = requests.post('http://127.0.0.1:8000/api/qa/ask', json=qa_req)
    qa_data = r_qa.json() if r_qa.status_code == 200 else {}
    total_time = round(time.time() - t0, 3)

    summary = {
        'category': tc['category'],
        'file': p.name,
        'anchor_c1': cap_data.get('selected_pair', {}).get('rank_1_caption', 'N/A'),
        'fused_caption': cap_data.get('fusion', {}).get('fused_caption', 'N/A'),
        'ocr_has_text': ocr_data.get('has_text', False),
        'ocr_text_snippet': ocr_data.get('text', '')[:100],
        'ocr_lines_count': len(ocr_data.get('lines', [])),
        'question': tc['question'],
        'qa_answer': qa_data.get('answer', 'N/A'),
        'qa_capability': qa_data.get('capability', 'N/A'),
        'qa_uncertain': qa_data.get('is_uncertain', False),
        'qa_sources': qa_data.get('grounded_sources', []),
        'latency_seconds': total_time
    }
    results.append(summary)
    print(f"=== {tc['category']} ({p.name}) ===")
    print(f"  Anchor: {summary['anchor_c1']}")
    print(f"  Fused:  {summary['fused_caption']}")
    print(f"  OCR:    has_text={summary['ocr_has_text']}, lines={summary['ocr_lines_count']}, snippet='{summary['ocr_text_snippet']}'")
    print(f"  Q:      {tc['question']}")
    print(f"  A:      {summary['qa_answer']}")
    print(f"  Capab:  {summary['qa_capability']} (Uncertain: {summary['qa_uncertain']}, Sources: {summary['qa_sources']})")
    print(f"  Time:   {total_time}s\n")

out_file = Path('evaluations/phase2_acceptance_results.json')
out_file.write_text(json.dumps(results, indent=2), encoding='utf-8')
print(f"Saved complete acceptance results to {out_file}")

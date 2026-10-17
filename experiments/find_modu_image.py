import json
from pathlib import Path

files = [
    Path('C:/Users/Admin/.gemini/antigravity/brain/da1fd613-a480-4df5-bf8a-79b8cfac25c8/.system_generated/logs/transcript.jsonl'),
    Path('C:/Users/Admin/.gemini/antigravity/brain/ea6dc6ab-053d-491e-ac7f-16e7fb8d7cd5/.system_generated/logs/transcript.jsonl'),
    Path('C:/Users/Admin/.gemini/antigravity/brain/e25ca8ba-591e-4546-8c5b-9c7bec059305/.system_generated/logs/transcript.jsonl'),
]

for fp in files:
    if not fp.exists():
        continue
    for line in fp.read_text(encoding='utf-8', errors='ignore').splitlines():
        if 'Modu' in line or 'modu' in line:
            try:
                obj = json.loads(line)
                conv = fp.parent.parent.name
                t = obj.get("type")
                media = obj.get("media", [])
                print(f"[{conv}] type={t} media={media}")
                c = str(obj.get("content", ""))
                # find occurrences of image or path in c
                for word in c.split():
                    if any(ext in word.lower() for ext in ['.png', '.jpg', '.jpeg']):
                        print("   image word:", word)
            except Exception:
                pass

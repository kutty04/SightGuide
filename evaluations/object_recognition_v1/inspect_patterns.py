import csv
import json
from pathlib import Path

csv_path = Path("evaluations/object_recognition_v1/runs/baseline_v1_full/baseline_results.csv")
with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total baseline rows: {len(rows)}")

# Examine OCR
ocr_pos = [r for r in rows if r["ocr_has_text"] == "True"]
print(f"Images with OCR text detected: {len(ocr_pos)} / {len(rows)}")

# Common words / objects extracted
entities_all = []
for r in rows:
    ents = [e.strip() for e in r["extracted_entities"].split(",") if e.strip()]
    entities_all.extend(ents)

from collections import Counter
counts = Counter(entities_all).most_common(20)
print("\nTop 20 Extracted Noun-Phrases / Entities:")
for ent, c in counts:
    print(f"  {ent}: {c}")

print("\nSample 15 Captions & Extracted Entities:")
for r in rows[:15]:
    fn = r["filename"]
    st = r["stratum"]
    fused = r["fused_caption"]
    ocr = r["ocr_text"]
    safe_fused = fused.encode('ascii', errors='backslashreplace').decode('ascii')
    safe_ocr = ocr.encode('ascii', errors='backslashreplace').decode('ascii')
    print(f"[{fn}] ({st}): {safe_fused}")
    if ocr:
        print(f"   OCR: \"{safe_ocr}\"")

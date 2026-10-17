import json

with open("evaluations/object_recognition_v1/runs/pilot_detector_eval_v1/detector_results.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        r = json.loads(line)
        img = r["image_id"]
        st = r["stratum"]
        cap = r["baseline_caption"]
        dets = [f"{d['label']} ({d['confidence']:.2f})" for d in r["detections_raw"]]
        det_str = ", ".join(dets) if dets else "None"
        print(f"[{i:02d}] {img} ({st}): {det_str}")
        safe_cap = cap.encode('ascii', errors='backslashreplace').decode('ascii')
        print(f"     Caption: \"{safe_cap}\"")

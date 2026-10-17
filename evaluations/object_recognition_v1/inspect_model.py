import onnxruntime as ort
from pathlib import Path
import json

model_path = Path("evaluations/object_recognition_v1/models/yolov8n.onnx")
if not model_path.exists():
    print(f"Error: {model_path} does not exist")
    exit(1)

session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

print("--- ONNX Model Inspection ---")
print("Providers active:", session.get_providers())

# Inputs
for i, inp in enumerate(session.get_inputs()):
    print(f"Input {i}: name='{inp.name}', shape={inp.shape}, type={inp.type}")

# Outputs
for i, out in enumerate(session.get_outputs()):
    print(f"Output {i}: name='{out.name}', shape={out.shape}, type={out.type}")

meta = session.get_modelmeta()
print("\n--- Model Metadata ---")
print(f"Description: {meta.description}")
print(f"Producer: {meta.producer_name} (version {meta.version})")
print(f"Custom Metadata Keys: {list(meta.custom_metadata_map.keys())}")

if "names" in meta.custom_metadata_map:
    raw_names = meta.custom_metadata_map["names"]
    # Usually a dict representation like {0: 'person', 1: 'bicycle', ...}
    try:
        import ast
        class_names = ast.literal_eval(raw_names)
        print(f"Total classes: {len(class_names)}")
        print("First 10 classes:", {k: class_names[k] for k in list(class_names.keys())[:10]})
    except Exception as e:
        print("Raw names string:", raw_names[:200])

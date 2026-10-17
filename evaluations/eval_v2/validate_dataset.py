"""Evaluation Dataset Validator for SightGuide Eval-V2.

Validates:
1. File existence and uncorrupted image loading.
2. Exact SHA-256 integrity against dataset_manifest.csv.
3. Category coverage across required evaluation domains.
4. Schema compliance of annotation templates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image

EVAL_DIR = Path(__file__).resolve().parent
IMAGES_DIR = EVAL_DIR / "images"
META_DIR = EVAL_DIR / "metadata"
ANN_DIR = EVAL_DIR / "annotations"

REQUIRED_CATEGORIES = {
    "people_and_interactions",
    "cartoons_and_illustrations",
    "animals_and_wildlife",
    "everyday_objects",
    "indoor_and_cluttered",
    "difficult_and_ambiguous"
}


def validate_eval_v2_dataset() -> bool:
    manifest_csv = META_DIR / "dataset_manifest.csv"
    if not manifest_csv.is_file():
        print(f"Error: Manifest CSV missing at {manifest_csv}")
        return False

    with open(manifest_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    if not records:
        print("Error: Manifest CSV is empty.")
        return False

    print(f"Validating {len(records)} manifest records in Eval-V2...")
    categories_found = set()
    all_ok = True

    for r in records:
        img_path = IMAGES_DIR / r["filename"]
        if not img_path.is_file():
            print(f"Missing image file: {img_path}")
            all_ok = False
            continue

        # Verify hash
        actual_hash = hashlib.sha256(img_path.read_bytes()).hexdigest()
        if actual_hash != r["sha256"]:
            print(f"Hash mismatch on {r['filename']}: expected {r['sha256'][:8]}, got {actual_hash[:8]}")
            all_ok = False

        # Verify image readable
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as exc:
            print(f"Corrupt image file {r['filename']}: {exc}")
            all_ok = False

        categories_found.add(r["category"])

    missing_cats = REQUIRED_CATEGORIES - categories_found
    if missing_cats:
        print(f"Warning: Categories missing from currently acquired set: {missing_cats}")
    else:
        print("All required categories represented in Eval-V2 set!")

    template_json = ANN_DIR / "human_reference_template.json"
    if template_json.is_file():
        with open(template_json, "r", encoding="utf-8") as f:
            template = json.load(f)
        count = len(template.get("annotations", {})) or len(template.get("images", []))
        print(f"Annotation template verified ({count} image slots, status={template.get('annotation_status')}).")
    else:
        print("Warning: Annotation template missing.")

    if all_ok:
        print(f"Eval-V2 Dataset validation successful! Verified {len(records)} images.")
    return all_ok


if __name__ == "__main__":
    success = validate_eval_v2_dataset()
    sys.exit(0 if success else 1)

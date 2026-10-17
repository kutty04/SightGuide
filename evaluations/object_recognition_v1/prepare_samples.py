"""Reproducible Stratified Sampling and Extraction for SightGuide V2 Baseline.

Inputs:
  Archive: C:\\Users\\Admin\\Downloads\\sample-images-main.zip

Outputs:
  Manifest: evaluations/object_recognition_v1/sample_manifest.json
  Extracted Images: evaluations/object_recognition_v1/images/

Protocol:
  - Seed: 42
  - Target size: 120 unique images
  - Strata: RGB landscape (70), RGB portrait (18), RGB square (2),
            Grayscale L landscape (22), Grayscale L portrait (7), Grayscale L square (1)
  - Duplicate SHA-256 detection and exclusion
  - Per-image extraction verification against SHA-256 hash
"""

import io
import json
import hashlib
import os
import random
from pathlib import Path
import zipfile
from PIL import Image

ARCHIVE_PATH = Path(r"C:\Users\Admin\Downloads\sample-images-main.zip")
OUTPUT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = OUTPUT_DIR / "images"
MANIFEST_PATH = OUTPUT_DIR / "sample_manifest.json"

ALLOCATION = {
    "RGB_landscape": 70,
    "RGB_portrait": 18,
    "RGB_square": 2,
    "L_landscape": 22,
    "L_portrait": 7,
    "L_square": 1,
}

SEED = 42


def generate_manifest_and_extract(seed: int = SEED) -> dict:
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Source archive not found: {ARCHIVE_PATH}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    strata: dict[str, list[dict]] = {k: [] for k in ALLOCATION}
    duplicate_count = 0

    print(f"[Prepare] Reading archive: {ARCHIVE_PATH}")
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as z:
        jpg_names = sorted([n for n in z.namelist() if n.lower().endswith(".jpg")])
        for name in jpg_names:
            data = z.read(name)
            h = hashlib.sha256(data).hexdigest()
            if h in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(h)

            with Image.open(io.BytesIO(data)) as img:
                m = img.mode
                w, h_dim = img.size
                orient = "landscape" if w > h_dim else ("portrait" if h_dim > w else "square")
                key = f"{m}_{orient}"
                if key in strata:
                    strata[key].append({
                        "archive_path": name,
                        "filename": Path(name).name,
                        "sha256": h,
                        "width": w,
                        "height": h_dim,
                        "color_mode": m,
                        "aspect_ratio": round(w / h_dim, 4),
                        "orientation": orient,
                        "stratum": key,
                        "file_size_bytes": len(data)
                    })

        print(f"[Prepare] Total unique images: {len(seen_hashes)} | Duplicates skipped: {duplicate_count}")

        # Deterministic stratified sampling
        rng = random.Random(seed)
        sampled_images = []
        for stratum_name, target_count in ALLOCATION.items():
            pool = strata[stratum_name]
            available_count = len(pool)
            if available_count < target_count:
                raise ValueError(
                    f"Insufficient images in stratum {stratum_name}: requested {target_count}, available {available_count}"
                )
            # Sort deterministically by sha256 before sampling
            pool.sort(key=lambda x: x["sha256"])
            chosen = rng.sample(pool, target_count)
            sampled_images.extend(chosen)
            print(f"  Stratum '{stratum_name}': sampled {target_count} / {available_count}")

        # Sort all sampled entries by filename for stable indexing
        sampled_images.sort(key=lambda x: x["filename"])

        # Extract only sampled files and verify hash
        print(f"[Prepare] Extracting {len(sampled_images)} images into {IMAGES_DIR}...")
        verified_count = 0
        for item in sampled_images:
            archive_path = item["archive_path"]
            data = z.read(archive_path)
            actual_hash = hashlib.sha256(data).hexdigest()
            if actual_hash != item["sha256"]:
                raise ValueError(f"Hash mismatch extracting {archive_path}!")

            out_file = IMAGES_DIR / item["filename"]
            out_file.write_bytes(data)

            # Re-verify written file
            written_hash = hashlib.sha256(out_file.read_bytes()).hexdigest()
            if written_hash != item["sha256"]:
                raise ValueError(f"Disk write integrity failure for {out_file.name}!")
            verified_count += 1

        print(f"[Prepare] Successfully extracted and verified {verified_count} images.")

        manifest_data = {
            "metadata": {
                "source_archive": str(ARCHIVE_PATH),
                "sampling_seed": seed,
                "total_samples": len(sampled_images),
                "strata_allocation": ALLOCATION,
                "duplicate_images_excluded": duplicate_count,
            },
            "samples": sampled_images
        }

        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        print(f"[Prepare] Manifest saved to: {MANIFEST_PATH}")
        return manifest_data


if __name__ == "__main__":
    generate_manifest_and_extract(seed=SEED)

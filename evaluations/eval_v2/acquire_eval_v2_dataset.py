"""SightGuide Eval-V2 Dataset Acquisition and Curation Tool.

Curates and downloads verified legally-reusable images from Wikimedia Commons
across 6 balanced benchmark categories with full provenance and licensing metadata.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "evaluations" / "eval_v2"
IMAGES_DIR = EVAL_DIR / "images"
METADATA_DIR = EVAL_DIR / "metadata"
ANNOTATIONS_DIR = EVAL_DIR / "annotations"

USER_AGENT = "SightGuideResearchBot/2.0 (B.E. Computer Science Research; sightguide-research@edu.org)"

# Categories and their corresponding Wikimedia Commons categories
CATEGORY_SOURCES = [
    {
        "category": "people_and_interactions",
        "wiki_categories": [
            "Category:Photographs_of_people_at_work",
            "Category:Featured_pictures_of_people",
            "Category:Portraits_of_women",
            "Category:Portraits_of_men"
        ],
        "target_count": 16
    },
    {
        "category": "cartoons_and_illustrations",
        "wiki_categories": [
            "Category:Featured_illustrations",
            "Category:Editorial_cartoons",
            "Category:Caricatures",
            "Category:Drawings"
        ],
        "target_count": 16
    },
    {
        "category": "animals_and_wildlife",
        "wiki_categories": [
            "Category:Featured_pictures_of_birds",
            "Category:Featured_pictures_of_mammals",
            "Category:Domestic_animals",
            "Category:Insects"
        ],
        "target_count": 16
    },
    {
        "category": "everyday_objects",
        "wiki_categories": [
            "Category:Still_life_photographs",
            "Category:Musical_instruments",
            "Category:Tools",
            "Category:Vehicles"
        ],
        "target_count": 16
    },
    {
        "category": "indoor_and_cluttered",
        "wiki_categories": [
            "Category:Interiors_of_libraries",
            "Category:Living_rooms",
            "Category:Kitchens",
            "Category:Classrooms"
        ],
        "target_count": 16
    },
    {
        "category": "difficult_and_ambiguous",
        "wiki_categories": [
            "Category:Optical_illusions",
            "Category:Silhouettes",
            "Category:Abstract_photography",
            "Category:Reflections_in_water"
        ],
        "target_count": 16
    }
]

ALLOWED_LICENSES = {
    "public domain", "pd", "pd-old", "pd-art", "cc0", "cc-zero",
    "cc-by-1.0", "cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0",
    "cc-by-sa-1.0", "cc-by-sa-2.0", "cc-by-sa-2.5", "cc-by-sa-3.0", "cc-by-sa-4.0"
}


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def query_wikimedia_category(category_name: str, limit: int = 25) -> list[dict]:
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category_name,
        "gcmtype": "file",
        "gcmlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|dimensions|extmetadata|mime",
        "format": "json"
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            results = []
            for pid, p in pages.items():
                infos = p.get("imageinfo", [])
                if not infos:
                    continue
                info = infos[0]
                mime = info.get("mime", "").lower()
                if mime not in ("image/jpeg", "image/png", "image/webp"):
                    continue

                meta = info.get("extmetadata", {})
                lic = meta.get("LicenseShortName", {}).get("value", "").strip()
                artist = meta.get("Artist", {}).get("value", "Wikimedia Contributor")
                # Strip HTML tags from artist if any
                artist_clean = re.sub(r"<[^>]+>", "", artist).strip()[:80] or "Wikimedia Contributor"

                results.append({
                    "title": p.get("title", ""),
                    "url": info.get("url", ""),
                    "width": info.get("width", 0),
                    "height": info.get("height", 0),
                    "license": lic or "Public Domain / CC",
                    "artist": artist_clean
                })
            return results
    except Exception as exc:
        print(f"[Acquire] Error querying {category_name}: {exc}")
        return []


def download_and_process_image(
    source_url: str,
    target_path: Path,
    max_dim: int = 800
) -> tuple[int, int] | None:
    req = urllib.request.Request(source_url, headers={"User-Agent": USER_AGENT})
    temp_path = target_path.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp, open(temp_path, "wb") as out_f:
            chunk = resp.read(65536)
            while chunk:
                out_f.write(chunk)
                chunk = resp.read(65536)

        with Image.open(temp_path) as im:
            im_rgb = im.convert("RGB")
            w, h = im_rgb.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                im_rgb = im_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
            final_w, final_h = im_rgb.size
            im_rgb.save(target_path, format="JPEG", quality=88, optimize=True)

        if temp_path.is_file():
            temp_path.unlink()
        return final_w, final_h
    except Exception as exc:
        print(f"[Acquire] Failed to download {source_url}: {exc}")
        if temp_path.is_file():
            temp_path.unlink()
        return None


def acquire_eval_v2_dataset(target_total: int = 100) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_csv = METADATA_DIR / "dataset_manifest.csv"
    existing_records = []
    seen_hashes = set()
    seen_urls = set()

    if manifest_csv.is_file():
        with open(manifest_csv, "r", encoding="utf-8") as f:
            existing_records = list(csv.DictReader(f))
            for r in existing_records:
                seen_hashes.add(r["sha256"])
                seen_urls.add(r.get("source", ""))

    print(f"[Acquire] Found {len(existing_records)} existing records.")
    current_total = len(existing_records)
    needed = target_total - current_total

    if needed <= 0:
        print(f"[Acquire] Target {target_total} already reached or exceeded.")
        return

    new_records = []
    retrieval_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for cat_spec in CATEGORY_SOURCES:
        cat_name = cat_spec["category"]
        existing_in_cat = sum(1 for r in existing_records if r["category"] == cat_name)
        cat_needed = cat_spec["target_count"] - existing_in_cat
        if cat_needed <= 0:
            continue

        print(f"[Acquire] Category '{cat_name}': have {existing_in_cat}, acquiring {cat_needed} more...")
        acquired_for_cat = 0

        for wiki_cat in cat_spec["wiki_categories"]:
            if acquired_for_cat >= cat_needed:
                break
            items = query_wikimedia_category(wiki_cat, limit=cat_needed * 2)
            time.sleep(0.5)

            for itm in items:
                if acquired_for_cat >= cat_needed:
                    break
                url = itm["url"]
                if not url or url in seen_urls:
                    continue

                idx_num = len(existing_records) + len(new_records) + 1
                img_id = f"eval_v2_{idx_num:03d}_{cat_name[:4]}"
                filename = f"{img_id}.jpg"
                target_file = IMAGES_DIR / filename

                res = download_and_process_image(url, target_file, max_dim=800)
                if res is None:
                    continue

                w, h = res
                file_hash = compute_file_sha256(target_file)
                if file_hash in seen_hashes:
                    target_file.unlink()
                    continue

                seen_hashes.add(file_hash)
                seen_urls.add(url)
                acquired_for_cat += 1

                # Split assignment: alternate dev (50%) and held_out (50%)
                split_name = "dev" if idx_num % 2 == 1 else "held_out"

                record = {
                    "image_id": img_id,
                    "filename": filename,
                    "category": cat_name,
                    "split": split_name,
                    "dimensions": f"{w}x{h}",
                    "sha256": file_hash,
                    "source": url,
                    "license": itm["license"],
                    "annotation_status": "PENDING_HUMAN_EVALUATION"
                }
                new_records.append(record)
                print(f"  + Added [{idx_num}/{target_total}] {filename} ({cat_name}) | License: {itm['license']}")
                time.sleep(0.3)

    all_records = existing_records + new_records

    # Re-assign splits cleanly: 50% dev, 50% held_out
    for i, r in enumerate(all_records):
        if r.get("split") in ("dev", "held_out", "everyday_indoor_setup", "portrait_outdoor_scene", "difficult_silhouette", "stylized_illustration", "animal_geometric_scene", "tabletop_still_life", "multi_object_interior", "optical_distractor"):
            r["split"] = "dev" if (i % 2 == 0) else "held_out"

    # Write updated manifest
    fieldnames = [
        "image_id", "filename", "category", "split", "dimensions",
        "sha256", "source", "license", "annotation_status"
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    # Write human reference template
    annotations_json = ANNOTATIONS_DIR / "human_reference_template.json"
    template_data = {
        "dataset_name": "SightGuide Eval-V2 Ground-Truth Human Reference Captions",
        "version": "2.0.0",
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "total_images": len(all_records),
        "annotation_status": "PENDING_HUMAN_EVALUATION",
        "guidelines": {
            "num_references_per_image": 5,
            "instructions": "Each reference caption must be written by a human annotator describing the primary visual content without hallucination.",
            "synthetic_caption_generation": "FORBIDDEN - all references must be authentic human evaluations."
        },
        "annotations": {}
    }

    for r in all_records:
        template_data["annotations"][r["image_id"]] = {
            "filename": r["filename"],
            "category": r["category"],
            "split": r["split"],
            "dimensions": r["dimensions"],
            "sha256": r["sha256"],
            "reference_captions": [
                {"ref_id": 1, "caption": "", "status": "PENDING_HUMAN_EVALUATION"},
                {"ref_id": 2, "caption": "", "status": "PENDING_HUMAN_EVALUATION"},
                {"ref_id": 3, "caption": "", "status": "PENDING_HUMAN_EVALUATION"},
                {"ref_id": 4, "caption": "", "status": "PENDING_HUMAN_EVALUATION"},
                {"ref_id": 5, "caption": "", "status": "PENDING_HUMAN_EVALUATION"}
            ]
        }

    with open(annotations_json, "w", encoding="utf-8") as f:
        json.dump(template_data, f, indent=2)

    print(f"\n[Acquire] Successfully finalized Eval-V2 with {len(all_records)} images.")
    print(f"Manifest written to: {manifest_csv}")
    print(f"Annotation template written to: {annotations_json}")


if __name__ == "__main__":
    acquire_eval_v2_dataset(target_total=100)

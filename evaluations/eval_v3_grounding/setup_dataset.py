"""Setup script for Eval V3 Visual Grounding Dataset."""

import json
import hashlib
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_V3 = ROOT / "evaluations" / "eval_v3_grounding"
IMAGES_DIR = EVAL_V3 / "images"
META_DIR = EVAL_V3 / "metadata"

def setup():
    # 1. Copy portrait image
    src_portrait = ROOT / "evaluations" / "test_portrait_woman.png"
    dst_portrait = IMAGES_DIR / "01_portrait_woman_blue_shirt.png"
    shutil.copy2(src_portrait, dst_portrait)

    # 2. Copy images from eval_v2
    src_v2 = ROOT / "evaluations" / "eval_v2" / "images"
    v2_mapping = [
        ("eval_cartoon_avatar.png", "02_cartoon_avatar.png"),
        ("eval_person_silhouette.png", "03_person_cropped_silhouette.png"),
        ("eval_domestic_animal.png", "04_domestic_cat_chair.png"),
        ("eval_everyday_objects.png", "05_everyday_coffee_cup.png"),
        ("eval_cluttered_desk.png", "06_indoor_cluttered_desk.png"),
        ("eval_ambiguous_distractor.png", "07_ambiguous_illustration.png"),
    ]

    for s_name, d_name in v2_mapping:
        s_path = src_v2 / s_name
        d_path = IMAGES_DIR / d_name
        if s_path.exists():
            shutil.copy2(s_path, d_path)

    # 3. Create image containing clear visible text
    dst_text = IMAGES_DIR / "08_storefront_visible_text.png"
    text_img = Image.new("RGB", (512, 384), color=(240, 235, 220))
    draw = ImageDraw.Draw(text_img)
    # Draw sign board
    draw.rectangle([(80, 40), (432, 140)], fill=(30, 45, 60), outline=(20, 20, 20), width=3)
    # Draw text
    draw.text((110, 75), "DOWNTOWN BAKERY & CAFE", fill=(255, 220, 100))
    # Draw coffee cup silhouette
    draw.ellipse([(200, 200), (312, 312)], fill=(180, 120, 80))
    draw.rectangle([(220, 240), (292, 320)], fill=(240, 240, 240), outline=(50, 50, 50), width=2)
    text_img.save(dst_text, format="PNG")

    # 4. Build rich manifest
    manifest_records = [
        {
            "id": "v3_001_portrait_woman",
            "filename": "01_portrait_woman_blue_shirt.png",
            "category": "cropped_portrait",
            "source": "User-supplied test asset (Session upload media_1791942186672_e3a0bdce.png)",
            "provenance": "Original user uploaded illustration",
            "license": "Evaluation test use",
            "visibly_supported": [
                "young woman with wavy dark/black hair",
                "light blue collared shirt",
                "pearl earrings",
                "thin chain necklace",
                "cropped bust/upper-torso portrait",
                "digital 3D avatar/illustration style",
                "neutral gray background"
            ],
            "known_unsupported_hallucinations": [
                "jeans", "blue jeans", "pants", "trousers", "skirts", "shoes", "legs",
                "in the sims game", "the sims family", "disney movie"
            ]
        },
        {
            "id": "v3_002_cartoon_avatar",
            "filename": "02_cartoon_avatar.png",
            "category": "illustration",
            "source": "SightGuide Eval V2 dataset",
            "provenance": "Locally synthesized digital avatar asset",
            "license": "Project internal CC-BY-4.0",
            "visibly_supported": [
                "cartoon character face",
                "expressive eyes",
                "digital illustration art style"
            ],
            "known_unsupported_hallucinations": [
                "specific movie titles", "licensed video game franchises", "photographic camera settings"
            ]
        },
        {
            "id": "v3_003_person_silhouette",
            "filename": "03_person_cropped_silhouette.png",
            "category": "partially_visible_people",
            "source": "SightGuide Eval V2 dataset",
            "provenance": "Locally synthesized silhouette asset",
            "license": "Project internal CC-BY-4.0",
            "visibly_supported": [
                "dark silhouette of a person",
                "profile outline"
            ],
            "known_unsupported_hallucinations": [
                "facial features", "specific clothing colors", "fabric textures", "eye color"
            ]
        },
        {
            "id": "v3_004_domestic_animal",
            "filename": "04_domestic_cat_chair.png",
            "category": "animals_surroundings",
            "source": "SightGuide Eval V2 dataset",
            "provenance": "Locally synthesized animal scene",
            "license": "Project internal CC-BY-4.0",
            "visibly_supported": [
                "domestic cat / feline",
                "indoor chair or surface"
            ],
            "known_unsupported_hallucinations": [
                "dog", "outdoor lawn", "leash", "wild predator context"
            ]
        },
        {
            "id": "v3_005_everyday_objects",
            "filename": "05_everyday_coffee_cup.png",
            "category": "everyday_objects",
            "source": "SightGuide Eval V2 dataset",
            "provenance": "Locally synthesized tabletop object asset",
            "license": "Project internal CC-BY-4.0",
            "visibly_supported": [
                "tabletop surface",
                "mug or cup",
                "simple domestic setup"
            ],
            "known_unsupported_hallucinations": [
                "people drinking", "restaurant kitchen", "busy crowd"
            ]
        },
        {
            "id": "v3_006_cluttered_desk",
            "filename": "06_indoor_cluttered_desk.png",
            "category": "indoor_multi_object",
            "source": "SightGuide Eval V2 dataset",
            "provenance": "Locally synthesized interior workspace asset",
            "license": "Project internal CC-BY-4.0",
            "visibly_supported": [
                "desk or table surface",
                "multiple desktop objects",
                "indoor lighting"
            ],
            "known_unsupported_hallucinations": [
                "office workers", "outdoor view", "specific high-end computer brands"
            ]
        },
        {
            "id": "v3_007_ambiguous_illustration",
            "filename": "07_ambiguous_illustration.png",
            "category": "ambiguous_abstract",
            "source": "SightGuide Eval V2 dataset",
            "provenance": "Locally synthesized abstract visual asset",
            "license": "Project internal CC-BY-4.0",
            "visibly_supported": [
                "geometric abstract shapes",
                "stylized color gradients"
            ],
            "known_unsupported_hallucinations": [
                "photorealistic human face", "landscape mountains", "real-world vehicles"
            ]
        },
        {
            "id": "v3_008_visible_text_sign",
            "filename": "08_storefront_visible_text.png",
            "category": "images_with_visible_text",
            "source": "SightGuide Eval V3 text benchmark generator",
            "provenance": "Locally rendered storefront signboard with explicit text",
            "license": "Project internal CC-BY-4.0",
            "visibly_supported": [
                "signboard with text 'DOWNTOWN BAKERY & CAFE'",
                "coffee cup symbol / illustration",
                "storefront palette"
            ],
            "known_unsupported_hallucinations": [
                "incorrectly transcribed shop names", "outdoor traffic", "pedestrians entering"
            ]
        }
    ]

    for rec in manifest_records:
        p = IMAGES_DIR / rec["filename"]
        if p.exists():
            data = p.read_bytes()
            rec["sha256"] = hashlib.sha256(data).hexdigest()
            img = Image.open(p)
            rec["dimensions"] = f"{img.size[0]}x{img.size[1]}"
            rec["file_size_bytes"] = len(data)

    manifest_path = META_DIR / "eval_v3_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "eval_v3_grounding",
            "description": "SightGuide V2 Grounding & Anti-Hallucination Evaluation Suite",
            "total_test_cases": len(manifest_records),
            "records": manifest_records
        }, f, indent=2)

    print(f"Eval V3 dataset ready: {len(manifest_records)} images created in {IMAGES_DIR}")
    print(f"Manifest written to: {manifest_path}")

if __name__ == "__main__":
    setup()

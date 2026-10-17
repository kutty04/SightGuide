"""SightGuide LLM Caption Fusion Service.

Implements LLM-based caption fusion following the Conservative V2 prompt policy
established in Celona et al. (Springer, 2025) replication.
"""

from __future__ import annotations

import re
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import sys

# Ensure backend directory is in sys.path for top-level 'app' resolution
_backend_dir = str(Path(__file__).resolve().parent.parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

try:
    from app.schemas import FusionResult
except ImportError:
    from ..schemas import FusionResult


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

CONSERVATIVE_V2_SYSTEM_PROMPT = """You are an accurate, visually grounded image-caption fusion assistant.
Combine the candidate captions into one concise, factual caption.
Rules:
1. Primary Visual Anchor: Caption 1 has the highest visual alignment to the image. Treat it as the factual foundation.
2. Complementary Details: Only include details from Caption 2 if they are directly corroborated, plausible, and describe visible scene attributes.
3. Anti-Extrapolation: Do not invent objects, actions, locations, identities, or attributes outside the image frame.
   - In portraits, headshots, or cropped upper-torso views, never assume or mention lower-body attire (e.g. jeans, pants, shoes) or out-of-frame settings.
   - Do not attribute generic artwork or avatars to proprietary game or movie franchises (e.g. The Sims, Disney) unless explicitly verified.
4. Conservative Conflict Resolution: If candidates conflict or if Caption 2 adds speculative out-of-frame claims, adhere strictly to Caption 1.
5. OCR Evidence: If verified readable text from the image is provided, incorporate it naturally into the caption when relevant. The OCR evidence may provide raw machine text and, when applicable, a separate normalized interpretation hypothesis. Treat normalized OCR as a hypothesis rather than guaranteed ground truth, and use natural phrasing rather than quoting corrupted OCR noise or glyphs verbatim. Never let text alone invent unseen physical objects.
6. Clean Output: Return exactly one factual caption without explanations or quotation marks."""

FUSION_MODEL = "openai/gpt-oss-20b"
FUSION_TEMPERATURE = 0.1
FUSION_MAX_TOKENS = 400
FUSION_PROMPT_VERSION = "grounded_conservative_v3"


def apply_grounding_safeguard(caption: str, anchor_context: str = "") -> str:
    """Sanitize out-of-frame clothing and proprietary franchise hallucinations while preserving genuine visual details.
    
    Preserves:
    - Upper-body clothing (e.g. 'wearing blue shirt')
    - Physical visible attributes (hair, eyes, face, expression, earrings, necklace)
    - Full-body clothing in wide/street shots where lower body is actually present
    - Genuine objects and environmental details
    
    Prunes:
    - Extrapolated lower-body clothing on upper-torso portraits / headshots
    - Fictional/commercial franchise attributions ('in the sims game') on generic artwork
    """
    if not caption or not isinstance(caption, str):
        return ""

    text = caption.strip().strip('"').strip("'")

    # Detect portrait / headshot / face framing from either caption or anchor
    context_text = f"{anchor_context} {text}".lower()
    is_portrait = bool(
        re.search(r'\b(face|portrait|headshot|head|bust|profile|close[\s-]up)\b', context_text)
    )
    has_full_body = bool(
        re.search(r'\b(full[\s-]body|standing|walking|running|whole[\s-]body|from head to toe)\b', context_text)
    )
    is_cropped_portrait = is_portrait and not has_full_body
    has_lower_body_in_anchor = bool(
        re.search(r'\b(jeans|pants|trousers|shorts|slacks|sweatpants|shoes|sneakers|boots)\b', anchor_context, re.IGNORECASE)
    )

    # Prune unobservable lower-body clothing if portrait framing or uncorroborated by anchor
    if is_cropped_portrait or (not has_full_body and not has_lower_body_in_anchor):
        # e.g. "wearing blue shirt and jeans" -> "wearing blue shirt"
        text = re.sub(
            r'(\b(?:wearing|in)\s+[^,]+?)\s+and\s+(?:(?:blue|black|white|dark|light|faded)\s+)?(?:jeans|pants|trousers|shorts|slacks|sweatpants)\b',
            r'\1',
            text,
            flags=re.IGNORECASE
        )
        # e.g. ", wearing jeans" or " and jeans" at end of clause
        text = re.sub(
            r'[,;\s]+(?:and\s+)?(?:wearing\s+)?(?:(?:blue|black|white|dark|light|faded)\s+)?(?:jeans|pants|trousers|shorts|slacks|shoes|sneakers|boots)\b',
            '',
            text,
            flags=re.IGNORECASE
        )

    # Prune speculative proprietary franchises for generic illustrations / avatars
    text = re.sub(r'\b(?:in|from)\s+the\s+sims(?:\s+game)?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bthe\s+sims\s+game\b', 'an animated character', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfor\s+the\s+disney\s+movie\b', 'in digital illustration style', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdisney\s+movie\b', 'animated style', text, flags=re.IGNORECASE)

    # Clean whitespace and trailing punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[,\s]+|[,\s]+$', '', text)
    return text


# ---------------------------------------------------------------------------
# COCO noun extraction (inline subset, avoids circular import with detector)
# ---------------------------------------------------------------------------
_COCO_QUERY_MAP: Dict[str, str] = {
    "person": "person", "people": "person", "man": "person", "woman": "person",
    "boy": "person", "girl": "person", "child": "person", "human": "person",
    "bicycle": "bicycle", "bike": "bicycle", "car": "car", "automobile": "car",
    "motorcycle": "motorcycle", "airplane": "airplane", "plane": "airplane",
    "bus": "bus", "train": "train", "truck": "truck", "boat": "boat",
    "bird": "bird", "cat": "cat", "dog": "dog", "horse": "horse",
    "sheep": "sheep", "cow": "cow", "elephant": "elephant", "bear": "bear",
    "zebra": "zebra", "giraffe": "giraffe", "backpack": "backpack",
    "umbrella": "umbrella", "handbag": "handbag", "suitcase": "suitcase",
    "skateboard": "skateboard", "surfboard": "surfboard", "frisbee": "frisbee",
    "skis": "skis", "snowboard": "snowboard", "kite": "kite",
    "bottle": "bottle", "cup": "cup", "bowl": "bowl", "banana": "banana",
    "apple": "apple", "sandwich": "sandwich", "pizza": "pizza", "cake": "cake",
    "chair": "chair", "couch": "couch", "sofa": "couch", "bed": "bed",
    "toilet": "toilet", "tv": "tv", "television": "tv", "laptop": "laptop",
    "computer": "laptop", "mouse": "mouse", "keyboard": "keyboard",
    "cell phone": "cell phone", "phone": "cell phone", "smartphone": "cell phone",
    "book": "book", "clock": "clock", "vase": "vase",
}

_COCO_CLASS_SET: Set[str] = set(_COCO_QUERY_MAP.values())


def _extract_coco_nouns(caption: str) -> Set[str]:
    """Extract COCO class names mentioned in a caption."""
    c = caption.lower()
    found: Set[str] = set()
    for term, cls in _COCO_QUERY_MAP.items():
        if re.search(r'\b' + re.escape(term) + r'\b', c):
            found.add(cls)
    return found


def validate_anchor_against_detections(
    anchor_caption: str,
    all_captions: List[str],
    detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate the BLIP anchor caption against YOLO detections.

    Returns a dict with keys:
      - demote: bool — True if the anchor should be replaced
      - alternate_caption: str | None — the next best caption if demote=True
      - reason: str — brief explanation for logging
      - detected_classes: set — classes found at any confidence
      - anchor_classes: set — COCO nouns mentioned in the anchor

    Demotion policy (all three conditions must hold):
      1. The anchor names at least one COCO-vocabulary object.
      2. That object is ABSENT from ALL detections (even at confidence >= 0.20).
      3. A DIFFERENT object IS present in detections (confidence >= 0.20).

    Safety constraints:
      - Zero detections → never demote (absence ≠ proof of absence).
      - Anchor names no COCO nouns → never demote.
      - No alternative caption supported by detections → never demote.
    """
    no_demotion = {
        "demote": False,
        "alternate_caption": None,
        "detected_classes": set(),
        "anchor_classes": set(),
        "reason": "not_evaluated",
    }

    if not detections:
        no_demotion["reason"] = "no_detections_cannot_contradict"
        return no_demotion

    anchor_classes = _extract_coco_nouns(anchor_caption)
    coco_anchor = anchor_classes & _COCO_CLASS_SET

    if not coco_anchor:
        no_demotion["anchor_classes"] = anchor_classes
        no_demotion["reason"] = "anchor_has_no_coco_nouns"
        return no_demotion

    # Build detected sets (at two thresholds)
    detected_low: Set[str] = {
        d["label"] for d in detections if d.get("confidence", 0.0) >= 0.20
    }

    if not detected_low:
        no_demotion["anchor_classes"] = anchor_classes
        no_demotion["reason"] = "no_detections_above_floor_threshold"
        return no_demotion

    anchor_subjects_absent = coco_anchor - detected_low
    conflicting_detections = detected_low - anchor_classes

    if not anchor_subjects_absent or not conflicting_detections:
        return {
            "demote": False,
            "alternate_caption": None,
            "detected_classes": detected_low,
            "anchor_classes": anchor_classes,
            "reason": "no_strong_contradiction",
        }

    # Find first alternative caption that mentions a detected class
    alternate: Optional[str] = None
    for cap in all_captions[1:]:
        cap_coco = _extract_coco_nouns(cap)
        if cap_coco & detected_low:
            alternate = cap
            break

    if alternate is None:
        return {
            "demote": False,
            "alternate_caption": None,
            "detected_classes": detected_low,
            "anchor_classes": anchor_classes,
            "reason": "no_supported_alternate_available",
        }

    print(
        f"[FusionService] Anchor demotion: anchor claims {anchor_subjects_absent}, "
        f"YOLO found {conflicting_detections}. Promoting: '{alternate[:60]}...'"
    )
    return {
        "demote": True,
        "alternate_caption": alternate,
        "detected_classes": detected_low,
        "anchor_classes": anchor_classes,
        "reason": (
            f"anchor_claims_{anchor_subjects_absent}_absent;"
            f"yolo_found_{conflicting_detections}"
        ),
    }


def build_user_prompt(
    caption_1: str,
    caption_2: str,
    ocr_text: Optional[str] = None,
    ocr_surface_hint: Optional[str] = None,
    normalized_ocr_text: Optional[str] = None,
) -> str:
    """Format the grounded two-caption fusion prompt with anchor priority and optional OCR evidence."""
    base = (
        f"Combine these two candidate captions for the same image into one factual sentence.\n"
        f"Primary Anchor (Caption 1): {caption_1}\n"
        f"Secondary Detail (Caption 2): {caption_2}"
    )

    surface = f" on {ocr_surface_hint}" if ocr_surface_hint else ""
    if normalized_ocr_text and normalized_ocr_text.strip() and ocr_text and ocr_text.strip() and normalized_ocr_text.strip() != ocr_text.strip():
        base += (
            f"\n\nVerified OCR Evidence (machine-recognized text visible{surface}):\n"
            f'Raw OCR: "{ocr_text.strip()}"\n'
            f'Normalized Interpretation: "{normalized_ocr_text.strip()}"\n'
            f"Incorporate this text naturally if it is relevant to the scene description."
        )
    elif ocr_text and ocr_text.strip():
        base += (
            f"\n\nVerified OCR Evidence (machine-recognized text visible{surface}): "
            f'"{ocr_text.strip()}"'
            f"\nIncorporate this text naturally if it is relevant to the scene description."
        )

    base += "\n\nGrounded Caption:"
    return base


def fuse_caption_pair(
    caption_1: str,
    caption_2: str,
    api_key: Optional[str] = None,
    max_retries: int = 2,
    ocr_text: Optional[str] = None,
    ocr_surface_hint: Optional[str] = None,
    normalized_ocr_text: Optional[str] = None,
) -> FusionResult:
    """Fuse two candidate captions using Groq LLM with Conservative V2 prompts.

    Args:
        caption_1: Primary visual anchor caption.
        caption_2: Secondary diversity caption.
        api_key: Optional Groq API key override.
        max_retries: Number of API retry attempts.
        ocr_text: Optional verified raw OCR text from the image.
        ocr_surface_hint: Optional surface description for OCR context (e.g. 'a phone screen').
        normalized_ocr_text: Optional normalized OCR hypothesis for noisy handwriting.

    Gracefully handles absence of API key by returning an informative
    status rather than raising an unhandled exception.
    """
    c1 = caption_1.strip()
    c2 = caption_2.strip()

    if not c1 and not c2:
        return FusionResult(
            fused_caption="",
            fusion_model=FUSION_MODEL,
            prompt_version=FUSION_PROMPT_VERSION,
            generation_status="FAILED",
            provenance="VALIDATION_ERROR",
            error_details="Both input captions are empty."
        )

    # Pre-sanitize candidate inputs using grounding safeguard
    c1_sanitized = apply_grounding_safeguard(c1, anchor_context=c1) or c1
    c2_sanitized = apply_grounding_safeguard(c2, anchor_context=c1_sanitized) or c2

    # Resolve Groq API key
    if api_key is not None:
        key = api_key.strip()
    else:
        key = os.environ.get("GROQ_API_KEY", "").strip()

    if not key:
        return FusionResult(
            fused_caption=c1_sanitized,
            fusion_model=FUSION_MODEL,
            prompt_version=FUSION_PROMPT_VERSION,
            generation_status="GROQ_KEY_NOT_CONFIGURED",
            provenance="UNEXECUTED_NO_CREDENTIAL",
            error_details="GROQ_API_KEY is not configured on the backend server. Set GROQ_API_KEY to enable live LLM fusion."
        )

    try:
        from groq import Groq
        client = Groq(api_key=key)
    except Exception as exc:
        return FusionResult(
            fused_caption="",
            fusion_model=FUSION_MODEL,
            prompt_version=FUSION_PROMPT_VERSION,
            generation_status="FAILED",
            provenance="CLIENT_INIT_ERROR",
            error_details=f"Failed to initialize Groq client: {exc}"
        )

    # Sanitize OCR text before injecting — preserve raw reading, no silent correction
    clean_ocr: Optional[str] = None
    if ocr_text and isinstance(ocr_text, str):
        stripped = ocr_text.strip()
        if len(stripped) >= 2:  # Ignore single-character noise
            clean_ocr = stripped

    clean_norm: Optional[str] = None
    if normalized_ocr_text and isinstance(normalized_ocr_text, str):
        stripped_norm = normalized_ocr_text.strip()
        if len(stripped_norm) >= 2:
            clean_norm = stripped_norm

    user_prompt = build_user_prompt(
        c1_sanitized, c2_sanitized,
        ocr_text=clean_ocr,
        ocr_surface_hint=ocr_surface_hint,
        normalized_ocr_text=clean_norm,
    )

    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=FUSION_MODEL,
                messages=[
                    {"role": "system", "content": CONSERVATIVE_V2_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=FUSION_MAX_TOKENS,
                temperature=FUSION_TEMPERATURE
            )
            content = response.choices[0].message.content
            if content and content.strip():
                clean_text = content.strip().strip('"')
                # Post-fusion verification: ensure LLM did not hallucinate ungrounded attire or brands
                final_grounded = apply_grounding_safeguard(clean_text, anchor_context=c1_sanitized) or c1_sanitized
                return FusionResult(
                    fused_caption=final_grounded,
                    fusion_model=FUSION_MODEL,
                    prompt_version=FUSION_PROMPT_VERSION,
                    generation_status="SUCCESS",
                    provenance="LIVE_GROQ_API_CALL",
                    error_details=""
                )
            last_error = "Model returned empty response text."
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(1)

    return FusionResult(
        fused_caption=c1_sanitized or c1,
        fusion_model=FUSION_MODEL,
        prompt_version=FUSION_PROMPT_VERSION,
        generation_status="FAILED",
        provenance="FALLBACK_ANCHOR_ON_API_ERROR",
        error_details=f"API calls failed after {max_retries} attempts: {last_error}"
    )

"""SightGuide LLM Caption Fusion Service.

Implements LLM-based caption fusion following the Conservative V2 prompt policy
established in Celona et al. (Springer, 2025) replication.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

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

CONSERVATIVE_V2_SYSTEM_PROMPT = """You are an accurate image-caption fusion assistant.
Combine the candidate captions into one concise, factual caption.
Rules:
1. Prefer details supported by the candidate captions.
2. Do not invent objects, actions, locations, identities, or attributes.
3. Do not add uncertain details merely to make the caption longer.
4. Preserve useful descriptive details when adequately supported.
5. If candidates conflict, prefer the more conservative description.
6. Return exactly one caption, without explanations or quotation marks."""

FUSION_MODEL = "openai/gpt-oss-20b"
FUSION_TEMPERATURE = 0.1
FUSION_MAX_TOKENS = 400
FUSION_PROMPT_VERSION = "conservative_v2"


def build_user_prompt(caption_1: str, caption_2: str) -> str:
    """Format the identical two-caption fusion prompt."""
    return f"""Combine these captions for the same image.
Caption 1: {caption_1}
Caption 2: {caption_2}
Write one accurate, concise caption."""


def fuse_caption_pair(
    caption_1: str,
    caption_2: str,
    api_key: Optional[str] = None,
    max_retries: int = 2
) -> FusionResult:
    """Fuse two candidate captions using Groq LLM with Conservative V2 prompts.

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

    # Resolve Groq API key
    key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()
    if not key:
        return FusionResult(
            fused_caption="",
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

    user_prompt = build_user_prompt(c1, c2)
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
                return FusionResult(
                    fused_caption=clean_text,
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
        fused_caption="",
        fusion_model=FUSION_MODEL,
        prompt_version=FUSION_PROMPT_VERSION,
        generation_status="FAILED",
        provenance="API_ERROR",
        error_details=f"API calls failed after {max_retries} attempts: {last_error}"
    )

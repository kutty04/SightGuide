"""SightGuide FastAPI Router Endpoints.

Provides RESTful API endpoints for:
1. Health and capability introspection (/api/health).
2. Image caption generation (/api/caption/generate).
3. Paired method comparison (/api/caption/compare).
4. Benchmark summary and image retrieval (/api/benchmark/*).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

import sys

# Ensure backend directory is in sys.path for top-level 'app' resolution
_backend_dir = str(Path(__file__).resolve().parent.parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

try:
    from app.schemas import (
        BenchmarkImageRecord,
        BenchmarkSummaryResponse,
        CaptionResponse,
        ComparisonResponse,
        HealthResponse,
        OCRResponse,
        QARequest,
        QAResponse,
    )
    from app.services.caption_pipeline import CaptionPipeline
    from app.services.ocr_service import OCRService
    from app.services.qa_service import QAService
except ImportError:
    from ..schemas import (
        BenchmarkImageRecord,
        BenchmarkSummaryResponse,
        CaptionResponse,
        ComparisonResponse,
        HealthResponse,
        OCRResponse,
        QARequest,
        QAResponse,
    )
    from ..services.caption_pipeline import CaptionPipeline
    from ..services.ocr_service import OCRService
    from ..services.qa_service import QAService


router = APIRouter(prefix="/api", tags=["SightGuide"])
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return system health, device info, model availability, OCR engine, and Groq configuration status."""
    pipeline = CaptionPipeline.get_instance()
    ocr_service = OCRService.get_instance()
    groq_set = bool(os.environ.get("GROQ_API_KEY", "").strip())
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        blip_available=pipeline.blip_loaded,
        ocr_available=ocr_service.is_available,
        groq_configured=groq_set,
        active_device=pipeline.device,
    )


@router.post("/caption/generate", response_model=CaptionResponse)
async def generate_caption(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    lambda_weight: float = Form(0.5, ge=0.0, le=1.0, description="MMR trade-off weight lambda"),
    method: str = Form("proposed", description="'control' or 'proposed'"),
    decoding_strategy: str = Form("sample", description="'sample' or 'beam' decoding strategy"),
) -> CaptionResponse:
    """Generate image caption using selected candidate ranking and fusion method."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size limit of {MAX_FILE_SIZE // (1024 * 1024)} MB."
        )

    pipeline = CaptionPipeline.get_instance()
    ocr_service = OCRService.get_instance()
    groq_key = os.environ.get("GROQ_API_KEY", "").strip() or None

    # Extract OCR text if engine is available so it can inform caption fusion
    extracted_ocr_text: Optional[str] = None
    extracted_norm_ocr_text: Optional[str] = None
    if ocr_service.is_available:
        try:
            ocr_res = await run_in_threadpool(ocr_service.extract_text, content)
            if ocr_res.get("has_text") and ocr_res.get("text"):
                extracted_ocr_text = ocr_res["text"].strip()
                if ocr_res.get("normalized_text"):
                    extracted_norm_ocr_text = ocr_res["normalized_text"].strip()
        except Exception as ocr_err:
            print(f"[Endpoints] Notice: OCR extraction skipped during captioning: {ocr_err}")

    try:
        result = await run_in_threadpool(
            pipeline.process_image,
            image_bytes=content,
            filename=file.filename or "uploaded_image.jpg",
            lambda_weight=lambda_weight,
            target_method=method.lower(),
            decoding_strategy=decoding_strategy.lower(),
            groq_api_key=groq_key,
            ocr_text=extracted_ocr_text,
            normalized_ocr_text=extracted_norm_ocr_text,
        )

        return CaptionResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Caption pipeline processing error: {exc}"
        )


@router.post("/caption/compare", response_model=ComparisonResponse)
async def compare_captions(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    lambda_weight: float = Form(0.5, ge=0.0, le=1.0, description="MMR trade-off weight lambda"),
    decoding_strategy: str = Form("sample", description="'sample' or 'beam' decoding strategy"),
) -> ComparisonResponse:
    """Run side-by-side comparison of Control vs. Diversity-Aware Proposed methods."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size limit of {MAX_FILE_SIZE // (1024 * 1024)} MB."
        )

    pipeline = CaptionPipeline.get_instance()
    groq_key = os.environ.get("GROQ_API_KEY", "").strip() or None

    try:
        result = await run_in_threadpool(
            pipeline.compare_methods,
            image_bytes=content,
            filename=file.filename or "uploaded_image.jpg",
            lambda_weight=lambda_weight,
            decoding_strategy=decoding_strategy.lower(),
            groq_api_key=groq_key
        )
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison pipeline error: {exc}"
        )


@router.get("/benchmark/summary", response_model=BenchmarkSummaryResponse)
async def get_benchmark_summary() -> BenchmarkSummaryResponse:
    """Retrieve verified experimental summary and per-image records from research CSV artifacts."""
    fc_path = PROJECT_ROOT / "results" / "sightguide_fusion_comparison.csv"
    if not fc_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Research comparison CSV artifact not found."
        )

    with open(fc_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ctl_rows = {
        r["image"]: r for r in rows
        if r["method"] == "CONTROL" and float(r["lambda_weight"]) == 1.0
    }
    prop_rows = {
        r["image"]: r for r in rows
        if r["method"] == "DIVERSITY_AWARE" and abs(float(r["lambda_weight"]) - 0.5) < 1e-4
    }

    records: List[BenchmarkImageRecord] = []
    identical_count = 0
    divergent_count = 0
    ctl_divs = []
    prop_divs = []
    failed_count = 0
    pending_count = 0

    for img in sorted(ctl_rows.keys()):
        c = ctl_rows[img]
        p = prop_rows.get(img, {})

        c_div = float(c.get("pair_diversity", 0.0))
        p_div = float(p.get("pair_diversity", 0.0))
        ctl_divs.append(c_div)
        prop_divs.append(p_div)

        is_div = p.get("diverged_from_control", "False").lower() == "true"
        if is_div:
            divergent_count += 1
        else:
            identical_count += 1

        p_status = p.get("generation_status", "")
        if "FAILED" in p_status:
            failed_count += 1
        elif "DRY_RUN" in p_status or "PENDING" in p_status:
            pending_count += 1

        records.append(BenchmarkImageRecord(
            image=img,
            image_path=f"/api/benchmark/image/{img}",
            control_c1=c.get("rank_1_caption", ""),
            control_c2=c.get("rank_2_caption", ""),
            control_fused=c.get("fused_caption", ""),
            proposed_c1=p.get("rank_1_caption", ""),
            proposed_c2=p.get("rank_2_caption", ""),
            proposed_fused=p.get("fused_caption", ""),
            is_divergent=is_div,
            control_pair_diversity=round(c_div, 4),
            proposed_pair_diversity=round(p_div, 4),
            status=p_status or "UNKNOWN"
        ))

    total = len(records)
    mean_ctl_div = round(sum(ctl_divs) / total, 4) if total else 0.0
    mean_prop_div = round(sum(prop_divs) / total, 4) if total else 0.0

    return BenchmarkSummaryResponse(
        total_images=total,
        identical_pairs_count=identical_count,
        divergent_pairs_count=divergent_count,
        control_mean_diversity=mean_ctl_div,
        proposed_mean_diversity=mean_prop_div,
        mean_diversity_delta=round(mean_prop_div - mean_ctl_div, 4),
        historical_failure_count=failed_count,
        pending_live_count=pending_count,
        benchmark_records=records
    )


@router.get("/benchmark/image/{image_name}")
async def get_benchmark_image(image_name: str) -> FileResponse:
    """Serve a local benchmark image from dataset directory."""
    clean_name = Path(image_name).name
    img_path = DATASET_DIR / clean_name
    if not img_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark image '{clean_name}' not found in dataset."
        )

    # Determine MIME type
    ext = img_path.suffix.lower()
    media_type = "image/jpeg"
    if ext == ".png":
        media_type = "image/png"
    elif ext == ".webp":
        media_type = "image/webp"

    return FileResponse(str(img_path), media_type=media_type)


@router.post("/ocr/extract", response_model=OCRResponse)
async def extract_image_text(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)")
) -> OCRResponse:
    """Extract visible text from image using local RapidOCR ONNX engine."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size limit of {MAX_FILE_SIZE // (1024 * 1024)} MB."
        )

    ocr_service = OCRService.get_instance()
    result = await run_in_threadpool(ocr_service.extract_text, content)
    return OCRResponse(
        filename=file.filename or "uploaded_image",
        text=result.get("text", ""),
        normalized_text=result.get("normalized_text"),
        has_text=result.get("has_text", False),
        lines=result.get("lines", []),
        message=result.get("message"),
        engine=result.get("engine", "rapidocr-onnxruntime-cpu"),
        execution_time_seconds=result.get("execution_time_seconds", 0.0)
    )



@router.get("/results/decoding-comparison")
async def get_decoding_comparison() -> dict:
    """Return latest versioned decoding comparison experiment records and summary metadata."""
    results_dir = PROJECT_ROOT / "evaluations" / "eval_v2" / "results"
    csv_file = results_dir / "eval_v2_decoding_comparison_v2.csv"
    json_file = results_dir / "eval_v2_experiment_metadata_v2.json"

    # Fall back to v1 if v2 not found
    if not csv_file.is_file():
        csv_file = results_dir / "eval_v2_decoding_comparison.csv"
        json_file = results_dir / "eval_v2_experiment_metadata.json"

    metadata = {}
    if json_file.is_file():
        import json
        with open(json_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    rows = []
    if csv_file.is_file():
        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    return {
        "metadata": metadata,
        "rows": rows,
        "total_records": len(rows),
        "source_file": csv_file.name if csv_file.is_file() else "none"
    }


@router.get("/results/manifest")
async def get_dataset_manifest() -> dict:
    """Return Eval-V2 dataset manifest summary and balanced category counts."""
    manifest_file = PROJECT_ROOT / "evaluations" / "eval_v2" / "metadata" / "dataset_manifest.csv"
    if not manifest_file.is_file():
        return {"total": 0, "categories": {}, "splits": {}, "records": []}

    records = []
    categories = {}
    splits = {}
    with open(manifest_file, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    for r in records:
        cat = r.get("category", "unknown")
        split = r.get("split", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        splits[split] = splits.get(split, 0) + 1

    return {
        "total_images": len(records),
        "categories": categories,
        "splits": splits,
        "records": records
    }


@router.post("/qa/ask", response_model=QAResponse)
async def ask_visual_question(
    payload: QARequest
) -> QAResponse:
    """Answer conversational questions grounded strictly in verified scene description and OCR text.

    Enforces honest uncertainty and strict grounding constraints.
    Rejects speculation about unobservable visual attributes.
    """
    qa_service = QAService.get_instance()
    response = await run_in_threadpool(qa_service.answer_question, payload)
    return response


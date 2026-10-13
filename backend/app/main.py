"""SightGuide Application Entrypoint.

FastAPI Application for Image Captioning, Diversity-Aware Ranking, and LLM Fusion.
Research foundation: Celona et al. (Springer, 2025).
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env into os.environ if present
_env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _k = _k.strip()
            _v = _v.strip().strip('"').strip("'")
            if _k not in os.environ:
                os.environ[_k] = _v

import sys

# Ensure backend directory is in sys.path for top-level 'app' resolution
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

try:
    from app.api.endpoints import router
    from app.services.caption_pipeline import CaptionPipeline
except ImportError:
    from .api.endpoints import router
    from .services.caption_pipeline import CaptionPipeline



@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown event handling."""
    print("=" * 60)
    print("SightGuide FastAPI Backend Starting...")
    # Initialize singleton caption pipeline
    pipeline = CaptionPipeline.get_instance()
    print(f"Device: {pipeline.device} | BLIP Available: {pipeline.blip_loaded}")
    print("=" * 60)
    yield
    print("SightGuide FastAPI Backend Shutting down...")


app = FastAPI(
    title="SightGuide API",
    description=(
        "Research and Application Backend for SightGuide: "
        "Extending Celona et al. (Springer 2025) with Diversity-Aware Caption-Pair Ranking "
        "and LLM-based Caption Fusion."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(router)


@app.get("/", tags=["Root"])
def root_info() -> dict[str, str]:
    """Root info endpoint providing API documentation link."""
    return {
        "service": "SightGuide Image Captioning & Fusion API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/api/health",
        "research_foundation": "Celona et al., Springer 2025"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

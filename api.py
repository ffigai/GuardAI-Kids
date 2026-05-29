"""FastAPI backend for GuardAI Kids Chrome extension."""
from __future__ import annotations

import logging
import os
import re
import sys
from typing import Literal

# Make src/guardaikids importable when running from the project root without
# pip-installing the package (no pyproject.toml / setup.py present).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import guardaikids.service as _svc  # noqa: E402
from guardaikids.service import analyze_youtube_url  # noqa: E402

# Cache loaded artifacts per (artifact_dir, mode) so model weights are read
# from disk only once per mode for the lifetime of the server process.
_artifacts_cache: dict[str, object] = {}
_original_load = _svc.load_analysis_artifacts

def _cached_load(artifact_dir=None, mode=None):
    key = f"{artifact_dir}:{mode}"
    if key not in _artifacts_cache:
        logger.info("Loading artifacts for mode=%s (first request — will be cached)", mode)
        _artifacts_cache[key] = _original_load(artifact_dir, mode)
    return _artifacts_cache[key]

_svc.load_analysis_artifacts = _cached_load

API_BASE_URL = "http://localhost:8000"

logger = logging.getLogger("guardai")

app = FastAPI(title="GuardAI Kids API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"(chrome-extension://.*|http://localhost(:\d+)?|https://(www\.)?youtube\.com)",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# In-memory cache keyed by YouTube video_id
_cache: dict[str, dict] = {}

_VIDEO_ID_RE = re.compile(r"[?&]v=([A-Za-z0-9_-]{11})")


def _extract_video_id(url: str) -> str | None:
    match = _VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


def _format_response(result: dict) -> dict:
    metadata = result.get("metadata", {})
    return {
        "verdict": result["decision"],
        "categories": result["categories"],
        "scores": result["model_scores"],
        "text_cues": result["top_tokens"],
        "image_cues": result["image_highlights"],
        "title": metadata.get("title", ""),
        "channel": metadata.get("channel", ""),
    }


class AnalyseRequest(BaseModel):
    url: str
    mode: Literal["text", "image", "multimodal"] = "text"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/cache/{video_id}")
def get_cached(video_id: str, mode: str = "text") -> dict:
    key = f"{video_id}:{mode}"
    if key not in _cache:
        raise HTTPException(status_code=404, detail="Video not in cache")
    return _cache[key]


@app.delete("/cache/{video_id}")
def clear_cached(video_id: str) -> dict:
    # Remove all mode variants for this video
    to_remove = [k for k in _cache if k.startswith(f"{video_id}:")]
    for k in to_remove:
        del _cache[k]
    return {"cleared": video_id}


@app.post("/analyse")
def analyse(request: AnalyseRequest) -> dict:
    video_id = _extract_video_id(request.url)
    if not video_id:
        raise HTTPException(status_code=422, detail="Could not extract video_id from URL")

    key = f"{video_id}:{request.mode}"
    if key in _cache:
        return _cache[key]

    api_key = os.environ.get("YOUTUBE_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="YOUTUBE_API_KEY is not set on the server")

    logger.info("Analysing url=%s mode=%s", request.url, request.mode)
    try:
        result = analyze_youtube_url(request.url, api_key, mode=request.mode)
    except ValueError as exc:
        logger.error("ValueError: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error analysing %s", request.url)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    formatted = _format_response(result)
    _cache[key] = formatted
    return formatted


# uvicorn api:app --host 0.0.0.0 --port 8000 --reload

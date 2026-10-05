"""FastAPI service that fronts the TrustFeed pipeline.

Endpoints cover the full editorial loop:

* ``POST /process`` - upload audio/video, get a transcript + clip candidates,
* ``POST /extract`` - same, but from an already-transcribed document (no ASR),
* ``GET  /jobs/{id}`` - fetch a job,
* ``POST /jobs/{id}/review`` - the **human-validation gate**: approve/reject clips
  and, on publish, receive a ready-to-post package. Nothing is "published"
  automatically - an editor signs off first.

Jobs live in an in-memory store: fine for a demo, and the obvious seam to swap for
a database in production. The extraction backend is chosen by configuration so the
same service runs the fast heuristic in CI and the local LLM on a GPU box.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from trustfeed import __version__, config
from trustfeed.api.schemas import (
    ExtractRequest,
    HealthResponse,
    JobResponse,
    PublishPackage,
    ReviewRequest,
    ReviewResponse,
)
from trustfeed.asr import Segment, Transcript
from trustfeed.extract.llm import extract
from trustfeed.extract.pipeline import process
from trustfeed.extract.schema import Extraction

logger = logging.getLogger(__name__)

# In-memory job store. Ephemeral by design; swap for a DB/queue in production.
_JOBS: dict[str, Extraction] = {}

app = FastAPI(
    title="TrustFeed API",
    version=__version__,
    summary="Turn foreign-language news audio into review-ready social clips.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("TRUSTFEED_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness probe for Docker and the orchestrator."""
    return HealthResponse(
        status="ok", version=__version__,
        asr_model=config.ASR_MODEL, llm_backend=config.LLM_BACKEND,
    )


@app.get("/config")
def get_config() -> dict[str, Any]:
    """Expose the operative configuration for the console."""
    return {
        "asr_model": config.ASR_MODEL,
        "llm_backend": config.LLM_BACKEND,
        "llm_model": config.LLM_MODEL,
        "max_clips": config.MAX_CLIPS,
        "distribution_channels": list(config.DISTRIBUTION_CHANNELS),
    }


@app.post("/extract", response_model=JobResponse)
def extract_endpoint(request: ExtractRequest) -> JobResponse:
    """Extract clip candidates from an already-transcribed document (no ASR)."""
    segments = [
        Segment(id=i, start=s.start, end=s.end, text=s.text)
        for i, s in enumerate(request.segments)
    ]
    transcript = Transcript(
        language=request.language, language_probability=1.0,
        duration=segments[-1].end, segments=segments,
    )
    extraction = extract(transcript)
    job_id = _store(extraction)
    return JobResponse(job_id=job_id, extraction=extraction)


@app.post("/process", response_model=JobResponse)
async def process_endpoint(file: UploadFile = File(...)) -> JobResponse:  # noqa: B008
    """Transcribe an uploaded media file and extract clip candidates."""
    suffix = Path(file.filename or "upload").suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp.close()
        extraction = process(tmp.name, log_mlflow=False)
    except Exception as err:  # noqa: BLE001 - surface a clean 400 to the client
        logger.exception("Processing failed")
        raise HTTPException(status_code=400, detail=f"Could not process file: {err}") from err
    finally:
        with suppress(OSError):
            os.unlink(tmp.name)
    job_id = _store(extraction)
    return JobResponse(job_id=job_id, extraction=extraction)


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    """Fetch a previously processed job."""
    return JobResponse(job_id=job_id, extraction=_require(job_id))


@app.post("/jobs/{job_id}/review", response_model=ReviewResponse)
def review_job(job_id: str, request: ReviewRequest) -> ReviewResponse:
    """Apply an editor's approve/reject decisions (the human-validation gate).

    Records each clip decision. When ``publish`` is set, the job is marked
    ``approved`` and a ready-to-post package is returned for the approved clips -
    the demo's stand-in for pushing to a social channel.
    """
    extraction = _require(job_id)
    for decision in request.decisions:
        if not 0 <= decision.index < len(extraction.clips):
            raise HTTPException(status_code=422, detail=f"No clip at index {decision.index}")
        extraction.clips[decision.index].approved = decision.approved

    published: list[PublishPackage] = []
    if request.publish:
        approved = extraction.approved_clips
        extraction.status = "approved" if approved else "rejected"
        published = [
            PublishPackage(
                timespan=c.timespan, caption=c.caption, quote=c.quote,
                channels=list(config.DISTRIBUTION_CHANNELS),
            )
            for c in approved
        ]
    _JOBS[job_id] = extraction
    return ReviewResponse(job_id=job_id, extraction=extraction, published=published)


def _store(extraction: Extraction) -> str:
    """Register an extraction under a fresh job id."""
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = extraction
    return job_id


def _require(job_id: str) -> Extraction:
    """Return a job or raise 404."""
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return _JOBS[job_id]


# Optionally serve a pre-built React bundle from this same container so one image
# hosts both API and console (e.g. on a single-port Space). Mounted last so API
# routes win; a no-op unless TRUSTFEED_STATIC_DIR points at a real directory.
_static_dir = os.getenv("TRUSTFEED_STATIC_DIR", "")
if _static_dir and Path(_static_dir).is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="frontend")
    logger.info("Serving frontend bundle from %s", _static_dir)

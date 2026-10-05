"""Pydantic request/response models for the TrustFeed API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from trustfeed.extract.schema import Extraction


class SegmentIn(BaseModel):
    """One transcript segment supplied to the text-only extraction endpoint."""

    start: float = Field(..., ge=0)
    end: float = Field(..., gt=0)
    text: str = Field(..., min_length=1)


class ExtractRequest(BaseModel):
    """Run extraction on an already-transcribed document (no ASR)."""

    language: str = Field("und", description="ISO language code, or 'und' if unknown.")
    segments: list[SegmentIn] = Field(..., min_length=1)


class ClipDecision(BaseModel):
    """An editor's approve/reject decision for one clip, by index."""

    index: int = Field(..., ge=0, description="Position of the clip in clips[].")
    approved: bool


class ReviewRequest(BaseModel):
    """The human-validation gate: per-clip decisions plus a final publish flag."""

    decisions: list[ClipDecision] = Field(default_factory=list)
    publish: bool = Field(
        default=False, description="If true, mark the job approved and package clips."
    )


class JobResponse(BaseModel):
    """A processing job: an extraction plus its identifier."""

    job_id: str
    extraction: Extraction


class PublishPackage(BaseModel):
    """A single ready-to-post item for an approved clip."""

    timespan: str
    caption: str
    quote: str
    channels: list[str]


class ReviewResponse(BaseModel):
    """Result of a review action: the updated job and any publish package."""

    job_id: str
    extraction: Extraction
    published: list[PublishPackage] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Liveness / readiness payload."""

    status: str
    version: str
    asr_model: str
    llm_backend: str

"""Typed models for the speech-to-text layer (Segment 4A)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TranscriptionResult(BaseModel):
    """Result of transcribing one audio file.

    ``language_probability`` is included only because faster-whisper provides
    it natively per call (``info.language_probability``); it is never invented.
    """

    text: str
    language: str = Field(..., description="ISO-639-1 code detected by the model (e.g. 'en', 'ur').")
    duration_seconds: float = Field(..., ge=0)
    processing_time_ms: float = Field(..., ge=0)
    language_probability: Optional[float] = Field(None, ge=0, le=1)

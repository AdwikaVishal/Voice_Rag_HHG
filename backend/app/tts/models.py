"""Typed models for the text-to-speech layer (Segment 4D)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TTSResult(BaseModel):
    """Result of one synthesis call (used internally / by scripts).

    ``audio_path`` points at a temporary WAV file owned by the caller — the
    caller must unlink it once the audio has been returned or copied. It is
    never exposed through the public API.
    """

    audio_path: str
    format: str = Field(..., description="Output container (wav).")
    language: Optional[str] = Field(
        None, description="Resolved language in script form (eng_Latn | urd_Arab)."
    )
    voice: Optional[str] = Field(None, description="TTS voice used (provider-specific id).")
    provider: str = Field("edge", description="TTS provider name.")
    model: str = Field("edge-tts", description="TTS provider/model identifier.")
    duration_seconds: float = Field(..., ge=0)
    processing_time_ms: float = Field(..., ge=0)


class TTSResponse(BaseModel):
    """Public TTS metadata block.

    Deliberately does NOT carry a filesystem path — audio is streamed by the
    API and the temp file is deleted right after the response is sent.
    """

    provider: str
    model: str
    voice: Optional[str] = None
    language: Optional[str] = Field(
        None, description="Resolved language in script form (eng_Latn | urd_Arab)."
    )
    format: str
    duration_seconds: float = Field(..., ge=0)
    processing_time_ms: float = Field(..., ge=0)

    @classmethod
    def from_result(cls, result: TTSResult) -> "TTSResponse":
        """Build the public block from an internal :class:`TTSResult`."""
        return cls(
            provider=result.provider,
            model=result.model,
            voice=result.voice,
            language=result.language,
            format=result.format,
            duration_seconds=result.duration_seconds,
            processing_time_ms=result.processing_time_ms,
        )

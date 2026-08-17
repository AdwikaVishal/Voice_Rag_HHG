"""Speech-to-text (Segment 4A): faster-whisper STT service.

Exposes :class:`STTService` — a lazy, process-wide-reusable faster-whisper
wrapper — and :data:`get_stt_service`, the singleton used by the API layer.
"""

from __future__ import annotations

from functools import lru_cache

from .models import TranscriptionResult
from .service import STTService

__all__ = ["STTService", "TranscriptionResult", "get_stt_service"]


@lru_cache(maxsize=1)
def get_stt_service() -> STTService:
    """Process-wide singleton STT service (model loads once, on first use)."""
    return STTService()

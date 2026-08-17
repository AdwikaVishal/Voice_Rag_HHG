"""Text-to-speech (Segment 4D): grounded answer -> audio.

Exposes :class:`TTSService` — the edge-tts (Microsoft Edge neural voices)
provider with PyAV MP3 -> WAV transcoding — and :data:`get_tts_service`, the
process-wide singleton used by the API layer.
"""

from __future__ import annotations

from functools import lru_cache

from .models import TTSResponse, TTSResult
from .service import SynthesisError, TTSError, TTSService, UnsupportedLanguageError

__all__ = [
    "TTSService",
    "TTSResult",
    "TTSResponse",
    "TTSError",
    "UnsupportedLanguageError",
    "SynthesisError",
    "get_tts_service",
]


@lru_cache(maxsize=1)
def get_tts_service() -> TTSService:
    """Process-wide singleton TTS service (stateless edge backend)."""
    return TTSService()

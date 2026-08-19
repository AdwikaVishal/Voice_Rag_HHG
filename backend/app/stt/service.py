"""Speech-to-text service -- provider-based router.

Selects between ``faster_whisper`` (local CPU model) and ``sarvam`` (cloud API)
via the ``STT_PROVIDER`` environment variable.  The public interface is
unchanged: :meth:`transcribe` accepts a file path and returns a
:class:`TranscriptionResult`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..config import STT_PROVIDER
from .models import TranscriptionResult

logger = logging.getLogger("stt")


class STTService:
    """Provider-based STT service that delegates to the configured backend."""

    def __init__(self) -> None:
        self.provider_name = STT_PROVIDER

        if self.provider_name == "sarvam":
            from .sarvam_provider import SarvamSTTProvider

            self.provider = SarvamSTTProvider()
            logger.info("STT service initialised with Sarvam")
        elif self.provider_name == "faster_whisper":
            from .faster_whisper_provider import FasterWhisperProvider

            self.provider = FasterWhisperProvider()
            logger.info("STT service initialised with Faster-Whisper")
        else:
            raise ValueError(f"Unknown STT provider: {self.provider_name!r}")

    @property
    def is_loaded(self) -> bool:
        if hasattr(self.provider, "is_loaded"):
            return self.provider.is_loaded
        return True  # cloud providers are always ready

    @property
    def load_count(self) -> int:
        if hasattr(self.provider, "load_count"):
            return self.provider.load_count
        return 0

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe *audio_path* and return text + detected language."""
        return self.provider.transcribe(audio_path, language=language, **kwargs)

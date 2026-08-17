"""Speech-to-text service (Segment 4A) — faster-whisper, CPU-only.

The service lazily loads a single faster-whisper model and reuses it for the
process lifetime — the model is never reloaded per request. Audio is decoded
by faster-whisper itself (bundled PyAV/ffmpeg), so ``.wav``, ``.mp3`` and
``.m4a`` all work without a system ffmpeg binary.

Whisper is multilingual: with no ``language`` hint the model detects the
spoken language per call and returns its ISO-639-1 code. Unicode is preserved
(e.g. Urdu is transcribed in Arabic script when the model decides the audio
is ``ur``).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

# ctranslate2 bundles its own OpenMP runtime; combined with torch / faiss /
# scikit-learn copies, macOS can abort on a duplicate runtime. Setting this
# before any OpenMP-using library is imported makes the process deterministic.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from faster_whisper import WhisperModel  # noqa: E402

from ..config import STT_COMPUTE_TYPE, STT_DEVICE, STT_MODEL_SIZE  # noqa: E402
from .models import TranscriptionResult  # noqa: E402

logger = logging.getLogger("stt")


class STTService:
    """Reusable faster-whisper wrapper with lazy model loading."""

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ) -> None:
        self.model_size = model_size or STT_MODEL_SIZE
        self.device = device or STT_DEVICE
        self.compute_type = compute_type or STT_COMPUTE_TYPE
        self._model: Optional[WhisperModel] = None
        self._load_count = 0

    # -- lifecycle --------------------------------------------------------
    def _ensure_loaded(self) -> WhisperModel:
        if self._model is None:
            logger.info(
                "Loading faster-whisper model size=%s device=%s compute_type=%s",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
            self._load_count += 1
        return self._model

    @property
    def is_loaded(self) -> bool:
        """Whether the Whisper model has been loaded yet."""
        return self._model is not None

    @property
    def load_count(self) -> int:
        """Number of times the model was loaded (must stay 1 in production)."""
        return self._load_count

    # -- transcription ----------------------------------------------------
    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe ``audio_path`` and return text + detected language.

        ``language`` optionally overrides the model's language detection (e.g.
        ``"ur"`` to force Urdu-script output for Hindustani speech that the
        detector would otherwise label ``hi``). Extra ``**kwargs`` (e.g.
        ``beam_size``, ``condition_on_previous_text``) are forwarded to
        faster-whisper.
        """
        model = self._ensure_loaded()
        start = time.perf_counter()
        segments, info = model.transcribe(str(audio_path), language=language, **kwargs)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        processing_ms = (time.perf_counter() - start) * 1000.0
        return TranscriptionResult(
            text=text,
            language=str(info.language or ""),
            duration_seconds=float(info.duration or 0.0),
            processing_time_ms=round(processing_ms, 2),
            language_probability=float(info.language_probability)
            if getattr(info, "language_probability", None) is not None
            else None,
        )

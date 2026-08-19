"""Faster-Whisper STT provider -- local CPU-only model (original default)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from faster_whisper import WhisperModel  # noqa: E402

from ..config import STT_COMPUTE_TYPE, STT_DEVICE, STT_MODEL_SIZE  # noqa: E402
from .models import TranscriptionResult  # noqa: E402

logger = logging.getLogger("stt")


class FasterWhisperProvider:
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

    def _ensure_loaded(self) -> WhisperModel:
        if self._model is None:
            logger.info(
                "Loading Faster-Whisper model %s on %s (%s)",
                self.model_size, self.device, self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
            self._load_count += 1
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_count(self) -> int:
        return self._load_count

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        model = self._ensure_loaded()
        start = time.perf_counter()
        segments, info = model.transcribe(
            str(audio_path), language=language or None, beam_size=5, **kwargs
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
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

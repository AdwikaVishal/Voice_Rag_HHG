"""Sarvam STT provider -- using official API endpoint from docs."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

from ..config import SARVAM_API_KEY, SARVAM_LANGUAGE, SARVAM_MODEL
from .models import TranscriptionResult

logger = logging.getLogger("stt")


class SarvamSTTProvider:
    """Sarvam Speech-to-Text using/speech-to-text endpoint.  Model: saaras:v3."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or SARVAM_API_KEY
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY environment variable is required")
        self.model = model or SARVAM_MODEL
        self.language = language or SARVAM_LANGUAGE
        self.endpoint = os.getenv(
            "SARVAM_ENDPOINT", "https://api.sarvam.ai/speech-to-text"
        )

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        path = Path(audio_path)
        headers = {"api-subscription-key": self.api_key}
        data = {"language": language or self.language, "model": self.model}
        start = time.perf_counter()

        try:
            with open(path, "rb") as fh:
                files = {"file": (path.name, fh, _guess_content_type(path))}
                resp = requests.post(
                    self.endpoint, headers=headers, files=files, data=data, timeout=30
                )
                resp.raise_for_status()
                result = resp.json()
        except requests.exceptions.RequestException as exc:
            logger.error("Sarvam API error: %s", exc)
            if hasattr(exc, "response") and exc.response is not None:
                logger.error("Status: %s", exc.response.status_code)
                logger.error("Body: %s", exc.response.text)
            raise RuntimeError(f"Sarvam STT failed: {exc}") from exc

        processing_ms = (time.perf_counter() - start) * 1000.0
        confidence = result.get("confidence")

        return TranscriptionResult(
            text=result.get("transcript") or result.get("text", ""),
            language=result.get("language", "en"),
            duration_seconds=float(result.get("duration", 0.0)),
            processing_time_ms=round(processing_ms, 2),
            language_probability=float(confidence) if confidence is not None else None,
        )


def _guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }.get(ext, "application/octet-stream")

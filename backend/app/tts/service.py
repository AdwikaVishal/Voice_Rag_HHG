"""Text-to-speech service (Segment 4D): grounded answer -> audio.

Provider: ``edge`` — Microsoft Edge neural voices via ``edge-tts``
(``en-US-AriaNeural`` for English, ``ur-PK-UzmaNeural`` for genuine Urdu). No
API key is required, but a network connection is. ``edge-tts`` streams MP3, so
the service transcodes it to a standard 24 kHz mono PCM WAV with PyAV (an
existing dependency of faster-whisper) — the browser/API-friendly format the
rest of the pipeline emits.

The service is stateless: each ``synthesize`` call writes a temporary WAV file
and returns its path; the caller owns that file and must delete it once the
audio has been returned. Intermediate MP3 files are always removed inside
``synthesize``, so no audio ever accumulates on disk.

Security: no credentials exist for this provider, nothing secret is ever
logged, and the output is never returned through the JSON API.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import av
import edge_tts

from ..config import TTS_FORMAT, TTS_PROVIDER, TTS_RATE, TTS_VOICE_EN, TTS_VOICE_UR
from .models import TTSResponse, TTSResult

logger = logging.getLogger("tts")

EDGE_PROVIDER = "edge"
TTS_MODEL = "edge-tts"
WAV_SAMPLE_RATE = 24000

# Language aliases -> (resolved script-form code, voice). Urdu maps to a
# genuine Urdu voice only; it is never silently replaced by a Hindi voice.
# A language that does not resolve is rejected cleanly (no fallback guessing).
_ENG_ALIASES = frozenset({"eng_latn", "eng-latn", "en", "eng", "english"})
_URD_ALIASES = frozenset({"urd_arab", "urd-arab", "ur", "urd", "urdu"})


class TTSError(RuntimeError):
    """Base class for text-to-speech failures; never exposes internals."""


class UnsupportedLanguageError(TTSError):
    """Raised when the requested language has no genuine TTS voice."""

    def __init__(self, language: str) -> None:
        super().__init__(
            f"Unsupported TTS language {language!r}. Supported: eng_Latn (English), urd_Arab (Urdu)."
        )


class SynthesisError(TTSError):
    """Raised when the provider fails to produce valid audio."""


def _transcode_to_wav(source: Path, target: Path, sample_rate: int = WAV_SAMPLE_RATE) -> float:
    """Decode ``source`` (MP3 or WAV) to a mono PCM WAV; return duration (s)."""
    container = av.open(str(source))
    out = av.open(str(target), "w")
    ostream = out.add_stream("pcm_s16le", rate=sample_rate)
    ostream.layout = "mono"
    samples = 0
    input_rate = float(sample_rate)
    try:
        for frame in container.decode(audio=0):
            input_rate = float(frame.sample_rate)
            for packet in ostream.encode(frame):
                out.mux(packet)
            samples += int(frame.samples)
        for packet in ostream.encode():
            out.mux(packet)
    finally:
        out.close()
        container.close()
    return samples / input_rate if input_rate else 0.0


class TTSService:
    """Reusable, stateless edge-tts wrapper with MP3 -> WAV transcoding.

    ``synthesizer`` is injectable for tests (a callable
    ``(text, voice, output_path) -> None``); production uses edge-tts.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        voice_en: Optional[str] = None,
        voice_ur: Optional[str] = None,
        rate: Optional[str] = None,
        output_format: Optional[str] = None,
        synthesizer: Optional[Callable] = None,
    ) -> None:
        self.provider_name = (provider or TTS_PROVIDER).strip().lower()
        if self.provider_name != EDGE_PROVIDER:
            raise ValueError(f"Unsupported TTS provider: {self.provider_name!r}")
        self.voice_en = voice_en or TTS_VOICE_EN
        self.voice_ur = voice_ur or TTS_VOICE_UR
        self.rate = rate or TTS_RATE
        self.output_format = (output_format or TTS_FORMAT).strip().lower()
        self._synthesizer = synthesizer or self._edge_synthesize

    @property
    def provider(self) -> str:
        return self.provider_name

    @property
    def model(self) -> str:
        return TTS_MODEL

    # -- provider backend --------------------------------------------------
    def _edge_synthesize(self, text: str, voice: str, output_path: Path) -> None:
        """Synthesize ``text`` with edge-tts into ``output_path`` (MP3)."""
        edge_tts.Communicate(text, voice=voice, rate=self.rate).save_sync(str(output_path))

    def _resolve_voice(self, language: Optional[str]) -> tuple[str, Optional[str]]:
        """Map a language value to ``(edge voice, script-form code)``."""
        if not language:
            return self.voice_en, None
        code = language.strip().lower()
        if code in _ENG_ALIASES:
            return self.voice_en, "eng_Latn"
        if code in _URD_ALIASES:
            return self.voice_ur, "urd_Arab"
        raise UnsupportedLanguageError(language)

    # -- synthesis ---------------------------------------------------------
    def synthesize(self, text: str, language: Optional[str] = None) -> TTSResult:
        """Synthesize ``text`` into a temporary WAV; return metadata + path.

        The caller owns the returned ``audio_path`` and must delete it after
        the audio has been served. Unicode is preserved; the exact ``text`` is
        spoken (TTS never invents or rewrites content).
        """
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")

        voice, resolved = self._resolve_voice(language)
        start = time.perf_counter()

        fd, wav_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        wav_path = Path(wav_name)
        fd, mp3_name = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        mp3_path = Path(mp3_name)

        try:
            self._synthesizer(text, voice, mp3_path)
            duration = _transcode_to_wav(mp3_path, wav_path)
            if not wav_path.exists() or wav_path.stat().st_size == 0:
                raise SynthesisError("TTS produced no audio.")
        except UnsupportedLanguageError:
            wav_path.unlink(missing_ok=True)
            raise
        except TTSError:
            wav_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            logger.error("TTS provider %s failed: %s", self.provider_name, exc)
            wav_path.unlink(missing_ok=True)
            raise SynthesisError("Text-to-speech synthesis failed.") from exc
        finally:
            mp3_path.unlink(missing_ok=True)

        processing_ms = (time.perf_counter() - start) * 1000.0
        return TTSResult(
            audio_path=str(wav_path),
            format=self.output_format,
            language=resolved,
            voice=voice,
            provider=self.provider_name,
            model=self.model,
            duration_seconds=round(duration, 3),
            processing_time_ms=round(processing_ms, 2),
        )

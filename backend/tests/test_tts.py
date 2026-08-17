"""Unit tests for the text-to-speech layer (Segment 4D).

The real edge-tts network backend is never touched: a fake ``synthesizer`` is
injected into :class:`TTSService`, and it writes a deterministic local WAV (or
MP3 via PyAV's offline libmp3lame encoder) so the full MP3 -> WAV transcode
path runs offline and deterministically.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import TTS_VOICE_EN, TTS_VOICE_UR  # noqa: E402
from app.tts import (  # noqa: E402
    SynthesisError,
    TTSResponse,
    TTSResult,
    TTSService,
    UnsupportedLanguageError,
)

RATE = 24000


def write_silence_wav(path, seconds=0.5, rate=RATE):
    """Write a valid mono PCM WAV of ``seconds`` of silence."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


def write_silence_mp3(path, seconds=0.5, rate=RATE):
    """Write a valid MP3 of ``seconds`` of silence with PyAV (offline)."""
    import av
    import numpy as np

    container = av.open(str(path), "w")
    stream = container.add_stream("mp3", rate=rate)
    stream.layout = "mono"
    samples = int(rate * seconds)
    data = np.zeros((1, samples), dtype="int16")
    frame = av.AudioFrame.from_ndarray(data, format="s16", layout="mono")
    frame.sample_rate = rate
    frame.pts = 0
    for pkt in stream.encode(frame):
        container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


class RecordingSynthesizer:
    """Fake provider backend: records calls, writes a local WAV "MP3"."""

    def __init__(self):
        self.calls: list[tuple[str, str, Path]] = []
        self.fail = None

    def __call__(self, text, voice, output_path):
        if self.fail is not None:
            raise self.fail
        self.calls.append((text, voice, Path(output_path)))
        write_silence_wav(Path(output_path), seconds=0.5)
        return None


class TestTTSServiceInit(unittest.TestCase):
    def test_initialization_defaults(self):
        svc = TTSService(synthesizer=RecordingSynthesizer())
        self.assertEqual(svc.provider, "edge")
        self.assertEqual(svc.model, "edge-tts")
        self.assertEqual(svc.voice_en, TTS_VOICE_EN)
        self.assertEqual(svc.voice_ur, TTS_VOICE_UR)
        self.assertEqual(svc.output_format, "wav")

    def test_initialization_accepts_overrides(self):
        svc = TTSService(
            provider="edge", voice_en="en-US-GuyNeural", voice_ur="ur-PK-AsadNeural",
            rate="+10%", output_format="wav", synthesizer=RecordingSynthesizer(),
        )
        self.assertEqual(svc.voice_en, "en-US-GuyNeural")
        self.assertEqual(svc.voice_ur, "ur-PK-AsadNeural")
        self.assertEqual(svc.rate, "+10%")

    def test_unsupported_provider_rejected_at_init(self):
        with self.assertRaises(ValueError):
            TTSService(provider="fake_provider")

    def test_get_tts_service_is_singleton(self):
        from app.tts import get_tts_service

        self.assertIs(get_tts_service(), get_tts_service())


class TestSynthesize(unittest.TestCase):
    def setUp(self):
        self.backend = RecordingSynthesizer()
        self.service = TTSService(synthesizer=self.backend)
        self._wavs: list[Path] = []

    def tearDown(self):
        for path in self._wavs:
            path.unlink(missing_ok=True)

    def _synth(self, *args, **kwargs) -> TTSResult:
        result = self.service.synthesize(*args, **kwargs)
        self._wavs.append(Path(result.audio_path))
        return result

    def _wav_duration(self, path: Path) -> float:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()

    def test_english_synthesis_request(self):
        result = self._synth("What is the capital of France?", language="eng_Latn")
        self.assertEqual(result.language, "eng_Latn")
        self.assertEqual(result.voice, TTS_VOICE_EN)
        self.assertEqual(result.format, "wav")
        self.assertEqual(result.provider, "edge")
        self.assertEqual(result.model, "edge-tts")
        text, voice, _ = self.backend.calls[0]
        self.assertEqual(text, "What is the capital of France?")
        self.assertEqual(voice, TTS_VOICE_EN)

    def test_urdu_synthesis_request(self):
        text = "فرانس کا دارالحکومت کیا ہے؟"
        result = self._synth(text, language="urd_Arab")
        self.assertEqual(result.language, "urd_Arab")
        self.assertEqual(result.voice, TTS_VOICE_UR)
        spoken, voice, _ = self.backend.calls[0]
        self.assertEqual(spoken, text)
        self.assertEqual(voice, TTS_VOICE_UR)

    def test_language_aliases(self):
        result = self._synth("x", language="en")
        self.assertEqual(result.language, "eng_Latn")
        result = self._synth("x", language="ur")
        self.assertEqual(result.language, "urd_Arab")

    def test_no_language_uses_default_english_voice(self):
        result = self._synth("Hello")
        self.assertIsNone(result.language)
        self.assertEqual(result.voice, TTS_VOICE_EN)

    def test_unsupported_language_handling(self):
        with self.assertRaises(UnsupportedLanguageError):
            self.service.synthesize("Namaste", language="hin_Deva")
        self.assertEqual(self.backend.calls, [])  # no provider call attempted

    def test_empty_text_handling(self):
        for bad in ("", "   ", "\n\t"):
            with self.assertRaises(ValueError):
                self.service.synthesize(bad, language="eng_Latn")
        self.assertEqual(self.backend.calls, [])

    def test_unicode_preservation(self):
        text = "日本語テスト — عربي — emoji 🎤 ے۔"
        result = self._synth(text, language="urd_Arab")
        spoken, _, _ = self.backend.calls[0]
        self.assertEqual(spoken, text)

    def test_output_file_validation(self):
        result = self._synth("Hello world", language="eng_Latn")
        path = Path(result.audio_path)
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 44)  # valid WAV header + data
        with wave.open(str(path), "rb") as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getframerate(), RATE)
        self.addCleanup(path.unlink, missing_ok=True)

    def test_duration_calculation(self):
        result = self._synth("Hello", language="eng_Latn")
        self.assertAlmostEqual(result.duration_seconds, 0.5, delta=0.05)
        self.assertEqual(result.duration_seconds, round(self._wav_duration(Path(result.audio_path)), 3))
        self.addCleanup(Path(result.audio_path).unlink, missing_ok=True)

    def test_latency_calculation(self):
        result = self._synth("Hello", language="eng_Latn")
        self.assertGreater(result.processing_time_ms, 0)
        self.assertEqual(result.processing_time_ms, round(result.processing_time_ms, 2))
        self.addCleanup(Path(result.audio_path).unlink, missing_ok=True)

    def test_intermediate_cleanup(self):
        result = self._synth("Hello", language="eng_Latn")
        # The intermediate provider artifact the fake was given is gone.
        _, _, intermediate = self.backend.calls[0]
        self.assertFalse(Path(intermediate).exists())
        # The caller-owned WAV survives until the caller deletes it.
        wav_path = Path(result.audio_path)
        self.assertTrue(wav_path.exists())
        wav_path.unlink()
        self.assertFalse(wav_path.exists())

    def test_no_temp_artifacts_after_failure(self):
        before = set(p.name for p in Path(tempfile.gettempdir()).glob("tmp*.wav"))
        self.backend.fail = RuntimeError("edge service down")
        with self.assertRaises(SynthesisError):
            self.service.synthesize("Hello", language="eng_Latn")
        after = set(p.name for p in Path(tempfile.gettempdir()).glob("tmp*.wav"))
        self.assertEqual(after, before)

    def test_provider_failure_handling(self):
        self.backend.fail = RuntimeError("network down")
        with self.assertRaises(SynthesisError) as ctx:
            self.service.synthesize("Hello", language="eng_Latn")
        self.assertNotIn("Traceback", str(ctx.exception))

    def test_response_schema(self):
        result = self._synth("Hello", language="eng_Latn")
        response = TTSResponse.from_result(result)
        data = response.model_dump()
        self.assertEqual(
            set(data),
            {"provider", "model", "voice", "language", "format", "duration_seconds", "processing_time_ms"},
        )
        self.assertEqual(data["provider"], "edge")
        self.assertEqual(data["model"], "edge-tts")
        self.assertEqual(data["voice"], TTS_VOICE_EN)
        self.assertEqual(data["language"], "eng_Latn")
        self.assertEqual(data["format"], "wav")
        self.assertGreater(data["duration_seconds"], 0)
        self.assertGreaterEqual(data["processing_time_ms"], 0)
        self.assertNotIn("audio_path", data)  # never exposes filesystem paths
        self.addCleanup(Path(result.audio_path).unlink, missing_ok=True)

    def test_real_offline_pipeline_with_mp3(self):
        """The full provider->transcode path with a genuine MP3 input (offline)."""
        backend = RecordingSynthesizer()
        original = backend.__call__

        def mp3_synthesizer(text, voice, output_path):
            write_silence_mp3(Path(output_path), seconds=0.5)
            return None

        backend.__call__ = mp3_synthesizer
        service = TTSService(synthesizer=backend)
        result = service.synthesize("Hello", language="eng_Latn")
        path = Path(result.audio_path)
        self.assertAlmostEqual(result.duration_seconds, 0.5, delta=0.15)  # mp3 codec delay
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 44)
        self.addCleanup(path.unlink, missing_ok=True)


if __name__ == "__main__":
    unittest.main()

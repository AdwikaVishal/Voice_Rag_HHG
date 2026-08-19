"""Unit tests for the speech-to-text layer (Segment 4A).

The real faster-whisper model is NEVER loaded here: ``WhisperModel`` is
patched with a fake so nothing is downloaded and no model is instantiated
unless a test drives it explicitly. Real-model checks live in
``scripts/test_stt.py`` (an opt-in integration script).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import STT_COMPUTE_TYPE, STT_DEVICE, STT_MODEL_SIZE  # noqa: E402
from app.stt import STTService, TranscriptionResult  # noqa: E402
from app.stt.faster_whisper_provider import FasterWhisperProvider  # noqa: E402

WAV_BYTES = b"RIFFfake-wav-bytes-for-tests"


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeInfo:
    def __init__(self, language="en", duration=1.5, language_probability=0.99):
        self.language = language
        self.duration = duration
        self.language_probability = language_probability


class FakeWhisperModel:
    """Patched ``WhisperModel``: records construction + transcribe arguments."""

    instances = 0
    language = "en"
    duration = 1.5
    language_probability = 0.99
    segments = [FakeSegment("hello world")]
    transcribe_error = None
    last_path = None
    last_language = None

    def __init__(self, model_size, device=None, compute_type=None):
        type(self).instances += 1
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, audio_path, language=None, **kwargs):
        if type(self).transcribe_error is not None:
            raise type(self).transcribe_error
        type(self).last_path = audio_path
        type(self).last_language = language
        info = FakeInfo(type(self).language, type(self).duration, type(self).language_probability)
        return iter(list(type(self).segments)), info


class TestSTTService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patcher = mock.patch("app.stt.faster_whisper_provider.WhisperModel", FakeWhisperModel)
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def setUp(self):
        FakeWhisperModel.instances = 0
        FakeWhisperModel.segments = [FakeSegment("hello world")]
        FakeWhisperModel.language = "en"
        FakeWhisperModel.duration = 1.5
        FakeWhisperModel.language_probability = 0.99
        FakeWhisperModel.transcribe_error = None
        FakeWhisperModel.last_path = None
        FakeWhisperModel.last_language = None

    def _make_provider(self, **kwargs) -> FasterWhisperProvider:
        return FasterWhisperProvider(**kwargs)

    def test_init_is_lazy(self):
        svc = STTService()
        self.assertFalse(svc.is_loaded)
        self.assertEqual(svc.load_count, 0)
        self.assertEqual(FakeWhisperModel.instances, 0)  # model not built yet

    def test_config_defaults(self):
        provider = self._make_provider()
        self.assertEqual(provider.model_size, STT_MODEL_SIZE)
        self.assertEqual(provider.device, STT_DEVICE)
        self.assertEqual(provider.compute_type, STT_COMPUTE_TYPE)

    def test_successful_transcription(self):
        svc = STTService()
        result = svc.transcribe("test.wav")
        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.duration_seconds, 1.5)
        self.assertGreaterEqual(result.processing_time_ms, 0)
        self.assertAlmostEqual(result.language_probability, 0.99)
        self.assertEqual(FakeWhisperModel.instances, 1)

    def test_model_loaded_once(self):
        svc = STTService()
        svc.transcribe("a.wav")
        svc.transcribe("b.wav")
        self.assertEqual(svc.load_count, 1)
        self.assertEqual(FakeWhisperModel.instances, 1)

    def test_language_extraction(self):
        FakeWhisperModel.language = "ur"
        result = STTService().transcribe("x.wav")
        self.assertEqual(result.language, "ur")

    def test_auto_detection_when_no_language_supplied(self):
        result = STTService().transcribe("en.wav")
        self.assertIsNone(FakeWhisperModel.last_language)  # model decides
        self.assertEqual(result.language, "en")  # actual detected code preserved

    def test_language_hint_reaches_transcription_layer(self):
        STTService().transcribe("x.wav", language="ur")
        self.assertEqual(FakeWhisperModel.last_language, "ur")

    def test_explicit_urdu_language_accepted(self):
        FakeWhisperModel.language = "ur"
        FakeWhisperModel.segments = [FakeSegment("فرانس کا دارالحکومت کیا ہے؟")]
        result = STTService().transcribe("urdu.wav", language="ur")
        self.assertEqual(FakeWhisperModel.last_language, "ur")
        self.assertEqual(result.language, "ur")
        self.assertEqual(result.text, "فرانس کا دارالحکومت کیا ہے؟")

    def test_automatic_detection_still_works_when_hint_given(self):
        FakeWhisperModel.language = "en"
        result = STTService().transcribe("en.wav", language="en")
        self.assertEqual(result.language, "en")

    def test_unicode_preserved(self):
        FakeWhisperModel.segments = [FakeSegment("فرانس کا دارالحکومت کیا ہے؟")]
        result = STTService().transcribe("urdu.wav")
        self.assertEqual(result.text, "فرانس کا دارالحکومت کیا ہے؟")

    def test_empty_transcription(self):
        FakeWhisperModel.segments = []
        result = STTService().transcribe("silence.wav")
        self.assertEqual(result.text, "")

    def test_multiple_segments_joined(self):
        FakeWhisperModel.segments = [FakeSegment("one."), FakeSegment("two.")]
        result = STTService().transcribe("multi.wav")
        self.assertEqual(result.text, "one. two.")

    def test_custom_constructor(self):
        provider = self._make_provider(model_size="base", device="cpu", compute_type="int8_float32")
        provider.transcribe("a.wav")
        self.assertEqual(provider.model_size, "base")
        self.assertEqual(provider.compute_type, "int8_float32")
        self.assertEqual(FakeWhisperModel.instances, 1)


class TestSTTAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patcher = mock.patch("app.stt.faster_whisper_provider.WhisperModel", FakeWhisperModel)
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def setUp(self):
        from fastapi.testclient import TestClient

        from app.main import app

        self.client = TestClient(app)
        FakeWhisperModel.instances = 0
        FakeWhisperModel.segments = [FakeSegment("hello world")]
        FakeWhisperModel.language = "en"
        FakeWhisperModel.duration = 1.5
        FakeWhisperModel.language_probability = 0.99
        FakeWhisperModel.transcribe_error = None
        FakeWhisperModel.last_path = None
        FakeWhisperModel.last_language = None

    def test_successful_transcription_schema(self):
        resp = self.client.post("/stt", files={"audio": ("test.wav", WAV_BYTES, "audio/wav")})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(set(body), {"text", "language", "duration_seconds", "processing_time_ms", "language_probability"})
        self.assertEqual(body["text"], "hello world")
        self.assertEqual(body["language"], "en")
        self.assertIsInstance(body["duration_seconds"], float)
        self.assertIsInstance(body["processing_time_ms"], float)

    def test_language_hint_optional_and_forwarded(self):
        resp = self.client.post(
            "/stt",
            files={"audio": ("test.wav", WAV_BYTES, "audio/wav")},
            data={"language": "ur"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(FakeWhisperModel.last_language, "ur")

    def test_no_language_hint_uses_auto_detection(self):
        resp = self.client.post("/stt", files={"audio": ("test.wav", WAV_BYTES, "audio/wav")})
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(FakeWhisperModel.last_language)
        self.assertEqual(resp.json()["language"], "en")

    def test_missing_file_rejected(self):
        resp = self.client.post("/stt")
        self.assertEqual(resp.status_code, 422)

    def test_unsupported_audio_type(self):
        resp = self.client.post("/stt", files={"audio": ("notes.txt", b"hello", "text/plain")})
        self.assertEqual(resp.status_code, 415)
        self.assertIn("Unsupported audio type", resp.json()["detail"])

    def test_empty_upload_rejected(self):
        resp = self.client.post("/stt", files={"audio": ("empty.wav", b"", "audio/wav")})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("empty", resp.json()["detail"].lower())

    def test_corrupt_audio_rejected(self):
        FakeWhisperModel.transcribe_error = ValueError("could not decode")
        resp = self.client.post("/stt", files={"audio": ("broken.wav", b"not-audio", "audio/wav")})
        self.assertEqual(resp.status_code, 422)
        self.assertNotIn("Traceback", resp.text)

    def test_transcription_failure_is_500_without_traceback(self):
        FakeWhisperModel.transcribe_error = RuntimeError("boom")
        resp = self.client.post("/stt", files={"audio": ("fail.wav", WAV_BYTES, "audio/wav")})
        self.assertEqual(resp.status_code, 500)
        self.assertNotIn("Traceback", resp.text)

    def test_unsupported_type_does_not_load_model(self):
        resp = self.client.post("/stt", files={"audio": ("notes.txt", b"x", "text/plain")})
        self.assertEqual(resp.status_code, 415)
        self.assertEqual(FakeWhisperModel.instances, 0)

    def test_temporary_file_cleaned_up(self):
        resp = self.client.post("/stt", files={"audio": ("cleanup.wav", WAV_BYTES, "audio/wav")})
        self.assertEqual(resp.status_code, 200)
        self.assertIsNotNone(FakeWhisperModel.last_path)
        self.assertFalse(Path(FakeWhisperModel.last_path).exists())

    def test_no_temp_files_left_in_tempdir(self):
        before = set(p.name for p in Path(tempfile.gettempdir()).glob("tmp*.wav"))
        resp = self.client.post("/stt", files={"audio": ("t.wav", WAV_BYTES, "audio/wav")})
        self.assertEqual(resp.status_code, 200)
        after = set(p.name for p in Path(tempfile.gettempdir()).glob("tmp*.wav"))
        self.assertEqual(after, before)

    def test_extension_case_insensitive(self):
        resp = self.client.post("/stt", files={"audio": ("UPPER.WAV", WAV_BYTES, "audio/wav")})
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()

"""Segment 4A — real speech-to-text integration test.

Loads the real faster-whisper model (downloaded on first run) and transcribes
one audio file, printing the transcript, detected language, duration,
processing time and real-time factor (RTF).

This is an OPT-IN integration script — it is NOT part of the unit-test suite,
so the normal ``unittest discover`` run never touches the Whisper model.

Usage (from backend/):
    python scripts/test_stt.py data/audio/test_english.wav
    python scripts/test_stt.py data/audio/test_urdu.wav --language ur
    python scripts/test_stt.py path/to/any.wav --model base --compute-type int8

``--language`` optionally overrides language detection (e.g. ``ur`` forces
Urdu-script output for Hindustani speech the detector would otherwise label
``hi``). ``--model`` / ``--compute-type`` override the config defaults for
this one run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import STT_COMPUTE_TYPE, STT_DEVICE, STT_MODEL_SIZE  # noqa: E402
from app.stt import STTService  # noqa: E402
from app.stt.faster_whisper_provider import FasterWhisperProvider  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path, help="path to the audio file (.wav/.mp3/.m4a)")
    parser.add_argument("--language", default=None, help="force language (e.g. en, ur)")
    parser.add_argument("--model", default=STT_MODEL_SIZE, help=f"faster-whisper size (default {STT_MODEL_SIZE})")
    parser.add_argument("--device", default=STT_DEVICE)
    parser.add_argument("--compute-type", default=STT_COMPUTE_TYPE)
    args = parser.parse_args()

    if not args.audio_path.exists():
        print(f"ERROR: file not found: {args.audio_path}", file=sys.stderr)
        raise SystemExit(1)

    load_start = time.perf_counter()
    provider = FasterWhisperProvider(model_size=args.model, device=args.device, compute_type=args.compute_type)
    print(f"Loading faster-whisper '{args.model}' (device={args.device}, "
          f"compute={args.compute_type}) ...")
    result = provider.transcribe(args.audio_path, language=args.language)
    load_ms = (time.perf_counter() - load_start) * 1000.0

    duration = result.duration_seconds
    processing_s = result.processing_time_ms / 1000.0
    rtf = processing_s / duration if duration > 0 else float("nan")

    print("\n--- Transcription ---")
    print(f"Audio:            {args.audio_path}")
    print(f"Language:         {result.language}"
          + (f" (p={result.language_probability:.3f})" if result.language_probability is not None else ""))
    print(f"Transcript:       {result.text}")
    print(f"Duration:         {duration:.2f} s")
    print(f"Processing time:  {result.processing_time_ms:.1f} ms "
          f"(model load {load_ms:.0f} ms once)")
    print(f"RTF:              {rtf:.2f}")

    if not result.text:
        print("\nWARNING: empty transcript (no speech detected).", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

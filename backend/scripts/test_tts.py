"""Segment 4D — real text-to-speech integration test.

Synthesizes one text into a WAV file with the real edge-tts backend (network
required) and prints the resolved language, output path, duration, processing
time and real-time factor (RTF = processing time / audio duration). The output
file is verified programmatically (exists, non-empty, valid WAV, duration > 0)
before the script exits.

This is an OPT-IN integration script — it is NOT part of the unit-test suite,
so the normal ``unittest discover`` run never makes a TTS network call.

Usage (from backend/):
    python scripts/test_tts.py --text "What is the capital of France?" --language eng_Latn
    python scripts/test_tts.py --text "فرانس کا دارالحکومت کیا ہے؟" --language urd_Arab
    python scripts/test_tts.py --text "Hello" --language en --output /tmp/out.wav

``--language`` accepts script-form codes (``eng_Latn`` / ``urd_Arab``) or
aliases (``en`` / ``ur``). The default output path is
``data/audio/tts_test_{english|urdu}.wav``.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.tts import TTSService, UnsupportedLanguageError  # noqa: E402

DEFAULT_OUTPUTS = {
    "eng_Latn": "data/audio/tts_test_english.wav",
    "urd_Arab": "data/audio/tts_test_urdu.wav",
}


def verify_wav(path: Path) -> float:
    """Verify a valid, non-empty WAV file; return its duration in seconds."""
    if not path.exists():
        raise RuntimeError(f"output file does not exist: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"output file is empty: {path}")
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            channels = wav.getnchannels()
            if frames <= 0 or rate <= 0 or channels <= 0:
                raise ValueError("empty/invalid WAV header")
    except (wave.Error, EOFError) as exc:
        raise RuntimeError(f"invalid WAV header: {exc}") from exc
    return frames / rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="text to synthesize (Unicode preserved)")
    parser.add_argument(
        "--language",
        default="eng_Latn",
        help="language: eng_Latn (default) or urd_Arab (aliases en/ur accepted)",
    )
    parser.add_argument("--output", default=None, help="override output WAV path")
    parser.add_argument(
        "--voice", default=None,
        help="override the edge voice (provider-specific id), e.g. en-US-GuyNeural",
    )
    args = parser.parse_args()

    language = args.language.strip().lower()
    code = "urd_Arab" if language in ("ur", "urd", "urdu", "urd_arab", "urd-arab") else "eng_Latn"
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = BASE_DIR / DEFAULT_OUTPUTS[code]

    print(f"TTS provider: edge-tts (Microsoft Edge neural voices)")
    print(f"Text:         {args.text}")
    print(f"Language:     {args.language}  (resolved {code})")

    service_kwargs: dict = {}
    if args.voice:
        service_kwargs["voice_ur" if code == "urd_Arab" else "voice_en"] = args.voice
    service = TTSService(**service_kwargs)
    if args.voice:
        print(f"Voice:        {args.voice} (override)")

    try:
        result = service.synthesize(args.text, language=code)
    except UnsupportedLanguageError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

    audio_path = Path(result.audio_path)
    try:
        duration = verify_wav(audio_path)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.replace(output_path)

    processing_s = result.processing_time_ms / 1000.0
    rtf = processing_s / duration if duration > 0 else float("nan")

    print(f"Output:       {output_path}")
    print(f"Duration:     {duration:.2f} s")
    print(f"Processing time: {result.processing_time_ms:.1f} ms")
    print(f"RTF:          {rtf:.2f}")

    if code == "urd_Arab" and result.language != "urd_Arab":
        print(f"\nWARNING: expected urd_Arab, got {result.language!r}.", file=sys.stderr)
        raise SystemExit(3)


if __name__ == "__main__":
    main()

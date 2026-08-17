"""Segment 4B–4D — real voice-query-pipeline integration test.

Runs the actual pipeline (real faster-whisper STT + real production hybrid
retriever + real Ollama LLM + real edge-tts TTS) on one audio file and prints
the transcript, guardrail verdict, resolved language, per-stage timings, the
top retrieved chunks, the grounded LLM answer and the TTS metadata.

This is an OPT-IN integration script — the unit suite never loads Whisper or
the embedding model and never calls the LLM or TTS providers.

Usage (from backend/):
    python scripts/test_voice_pipeline.py data/audio/test_english.wav
    python scripts/test_voice_pipeline.py data/audio/test_urdu.wav --language ur
    python scripts/test_voice_pipeline.py path/to/any.wav --top-k 5
    python scripts/test_voice_pipeline.py path/to/any.wav --no-tts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.pipeline import get_query_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path, help="path to the audio file (.wav/.mp3/.m4a)")
    parser.add_argument("--language", default=None, help="language hint (e.g. ur)")
    parser.add_argument("--top-k", type=int, default=5, help="number of chunks to retrieve")
    parser.add_argument("--no-tts", action="store_true", help="skip the TTS stage")
    args = parser.parse_args()

    if not args.audio_path.exists():
        print(f"ERROR: file not found: {args.audio_path}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Running voice pipeline on {args.audio_path} "
          f"(language={args.language!r}, top_k={args.top_k}, tts={'off' if args.no_tts else 'on'}) ...")

    response = get_query_pipeline().process_audio(
        args.audio_path, language_hint=args.language, top_k=args.top_k, with_tts=not args.no_tts
    )

    print("\n--- Voice query pipeline ---")
    print(f"Transcript:        {response.transcript}")
    print(f"Language:          {response.language}")
    print(f"Guardrail allowed: {response.guardrail.allowed} "
          f"(reason={response.guardrail.reason})")
    print(f"Timings:           stt={response.timings.stt_ms:.1f} ms | "
          f"guardrail={response.timings.guardrail_ms:.3f} ms | "
          f"retrieval={response.timings.retrieval_ms:.1f} ms | "
          f"llm={response.timings.llm_ms:.1f} ms | "
          f"tts={response.timings.tts_ms:.1f} ms | "
          f"total={response.timings.total_ms:.1f} ms")

    if response.retrieval is None:
        print("\nNo retrieval executed (input rejected by guardrail).")
        return

    ret = response.retrieval
    print(f"\n--- Retrieval (strategy={ret.strategy}, top_k={ret.top_k}, "
          f"chunks={ret.index_chunks}) ---")
    for i, r in enumerate(ret.results, start=1):
        print(f"{i}. [{r.chunk_id} | score={r.score} | {r.language}] {r.text[:120]}")

    if response.generation is not None:
        gen = response.generation
        print(f"\n--- Grounded answer ---")
        print(f"Answer:   {gen.answer}")
        print(f"Language: {gen.language} | grounded={gen.grounded} | "
              f"abstained={gen.abstained} | model={gen.model} | "
              f"context_count={gen.context_count} | latency={gen.latency_ms:.1f} ms")

    if response.tts is not None:
        tts = response.tts
        print(f"\n--- TTS (provider={tts.provider}, model={tts.model}) ---")
        print(f"Voice:   {tts.voice}")
        print(f"Language: {tts.language} | format={tts.format}")
        print(f"Duration: {tts.duration_seconds:.2f} s | processing={tts.processing_time_ms:.1f} ms "
              f"| RTF={tts.processing_time_ms / 1000.0 / tts.duration_seconds if tts.duration_seconds else float('nan'):.2f}")


if __name__ == "__main__":
    main()

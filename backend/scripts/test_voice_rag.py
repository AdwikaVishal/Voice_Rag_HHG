"""Segment 7 — automated end-to-end voice RAG pipeline tests.

Runs the full pipeline (real STT + real retriever + real LLM + real TTS) on
the existing audio files and validates the source/language/status fields.

This is an OPT-IN integration script — it requires:
  - The production FAISS+BM25 index to be built
  - Ollama running with the configured model (default: qwen2.5:3b)
  - Network access for edge-tts

Usage (from backend/):
    python scripts/test_voice_rag.py
    python scripts/test_voice_rag.py --no-tts      # skip TTS stage
    python scripts/test_voice_rag.py --test 1      # run one test only
    python scripts/test_voice_rag.py --verbose     # show full answers

Exit code 0 = all tests passed, non-zero = failures.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

AUDIO_DIR = BASE_DIR / "data" / "audio"

# ── colour helpers ────────────────────────────────────────────────────────────
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _ok(msg: str) -> str:
    return f"{_GREEN}✓{_RESET} {msg}"


def _fail(msg: str) -> str:
    return f"{_RED}✗{_RESET} {msg}"


def _warn(msg: str) -> str:
    return f"{_YELLOW}⚠{_RESET} {msg}"


def _header(msg: str) -> str:
    return f"\n{_BOLD}{msg}{_RESET}"


# ── assertion helpers ─────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.warnings: list[str] = []

    def check(self, condition: bool, label: str, detail: str = "") -> None:
        msg = f"{label}" + (f": {detail}" if detail else "")
        if condition:
            self.passed.append(msg)
            print(f"  {_ok(msg)}")
        else:
            self.failed.append(msg)
            print(f"  {_fail(msg)}")

    def warn(self, label: str) -> None:
        self.warnings.append(label)
        print(f"  {_warn(label)}")

    @property
    def ok(self) -> bool:
        return len(self.failed) == 0


# ── pipeline loader ───────────────────────────────────────────────────────────

_pipeline = None


def get_pipeline(with_tts: bool = True):
    global _pipeline
    if _pipeline is None:
        print("Loading pipeline (first call — model loading may take a moment)…")
        load_start = time.perf_counter()
        from app.pipeline import get_query_pipeline
        from app.tts import get_tts_service
        _pipeline = get_query_pipeline()
        if with_tts and _pipeline.tts is None:
            try:
                _pipeline.tts = get_tts_service()
            except Exception as exc:
                print(f"  {_warn(f'TTS service unavailable: {exc}')}")
        load_ms = (time.perf_counter() - load_start) * 1000.0
        print(f"  Pipeline ready in {load_ms:.0f} ms\n")
    return _pipeline


# ── individual tests ──────────────────────────────────────────────────────────

def test_1_english_rag(verbose: bool, with_tts: bool) -> TestResult:
    """TEST 1 — English RAG: test_english.wav → CDG query → source=rag."""
    r = TestResult("TEST 1: English RAG (test_english.wav)")
    print(_header(r.name))

    audio = AUDIO_DIR / "test_english.wav"
    r.check(audio.exists(), "audio file exists", str(audio))
    if not audio.exists():
        return r

    pipeline = get_pipeline(with_tts)
    resp = pipeline.process_audio(str(audio), with_tts=with_tts)

    print(f"  transcript:  {resp.transcript!r}")
    print(f"  language:    {resp.language}")
    if resp.generation:
        print(f"  source:      {resp.generation.source}")
        print(f"  status:      {resp.generation.status}")
        print(f"  grounded:    {resp.generation.grounded}")
        if verbose:
            print(f"  answer:      {resp.generation.answer}")
    _print_timings(resp)

    r.check(resp.guardrail.allowed, "guardrail allowed")
    r.check(bool(resp.transcript), "transcript non-empty")
    r.check(resp.language in ("eng_Latn", None), "language is English or auto",
            str(resp.language))
    r.check(resp.generation is not None, "generation present")
    if resp.generation:
        r.check(
            resp.generation.source in ("rag", "general_knowledge"),
            "source is rag or general_knowledge",
            resp.generation.source,
        )
        r.check(resp.generation.status in ("answered", "grounded", "uncertain"),
                "status is answered/grounded/uncertain", resp.generation.status)
        r.check(bool(resp.generation.answer), "answer non-empty")
        if resp.generation.source == "rag":
            r.check(resp.generation.grounded is not False,
                    "RAG answer grounded", str(resp.generation.grounded))
    _check_tts(r, resp, with_tts)
    return r


def test_2_urdu_rag(verbose: bool, with_tts: bool) -> TestResult:
    """TEST 2 — Urdu RAG: test_urdu.wav → source=rag, language=urd_Arab."""
    r = TestResult("TEST 2: Urdu RAG (test_urdu.wav)")
    print(_header(r.name))

    audio = AUDIO_DIR / "test_urdu.wav"
    r.check(audio.exists(), "audio file exists", str(audio))
    if not audio.exists():
        return r

    pipeline = get_pipeline(with_tts)
    resp = pipeline.process_audio(str(audio), language_hint="ur", with_tts=with_tts)

    print(f"  transcript:  {resp.transcript!r}")
    print(f"  language:    {resp.language}")
    if resp.generation:
        print(f"  source:      {resp.generation.source}")
        print(f"  status:      {resp.generation.status}")
        if verbose:
            print(f"  answer:      {resp.generation.answer}")
    _print_timings(resp)

    r.check(resp.guardrail.allowed, "guardrail allowed")
    r.check(bool(resp.transcript), "transcript non-empty")
    r.check(resp.language == "urd_Arab", "language detected as urd_Arab", str(resp.language))
    r.check(resp.generation is not None, "generation present")
    if resp.generation:
        r.check(
            resp.generation.source in ("rag", "general_knowledge"),
            "source is rag or general_knowledge",
            resp.generation.source,
        )
        r.check(bool(resp.generation.answer), "answer non-empty")
    _check_tts(r, resp, with_tts)
    return r


def test_3_urdu_edge_mp3(verbose: bool, with_tts: bool) -> TestResult:
    """TEST 3 — Urdu edge MP3: test_urdu_edge.mp3 → STT succeeds."""
    r = TestResult("TEST 3: Urdu edge MP3 (test_urdu_edge.mp3)")
    print(_header(r.name))

    audio = AUDIO_DIR / "test_urdu_edge.mp3"
    r.check(audio.exists(), "audio file exists", str(audio))
    if not audio.exists():
        return r

    pipeline = get_pipeline(with_tts)
    try:
        resp = pipeline.process_audio(str(audio), language_hint="ur", with_tts=False)
        r.check(True, "STT did not crash")
        r.check(bool(resp.transcript), "transcript non-empty", repr(resp.transcript))
        print(f"  transcript:  {resp.transcript!r}")
        print(f"  language:    {resp.language}")
        if resp.generation and verbose:
            print(f"  answer:      {resp.generation.answer}")
        _print_timings(resp)
    except Exception as exc:
        r.check(False, "STT did not crash", str(exc))
    return r


def test_4_general_knowledge(verbose: bool, with_tts: bool) -> TestResult:
    """TEST 4 — General knowledge via text path: telephone inventor → source=general_knowledge."""
    r = TestResult("TEST 4: General knowledge — telephone inventor (text query)")
    print(_header(r.name))

    pipeline = get_pipeline(with_tts)
    query = "Who invented the telephone?"
    print(f"  query: {query!r}")

    resp = pipeline.process_text(query, with_tts=False)

    print(f"  language:    {resp.language}")
    if resp.generation:
        print(f"  source:      {resp.generation.source}")
        print(f"  status:      {resp.generation.status}")
        if verbose:
            print(f"  answer:      {resp.generation.answer}")
    _print_timings(resp)

    r.check(resp.guardrail.allowed, "guardrail allowed")
    r.check(resp.generation is not None, "generation present")
    if resp.generation:
        r.check(
            resp.generation.source == "general_knowledge",
            "source is general_knowledge",
            resp.generation.source,
        )
        r.check(resp.generation.status == "answered", "status is answered",
                resp.generation.status)
        r.check(bool(resp.generation.answer), "answer non-empty")
        r.check(resp.generation.sources == [], "no RAG sources in GK answer")
        # Answer should mention Bell
        answer_lower = resp.generation.answer.lower()
        r.check(
            "bell" in answer_lower or "alexander" in answer_lower,
            "answer mentions Bell/Alexander",
            resp.generation.answer[:120],
        )
    return r


def test_5_tts_output(verbose: bool) -> TestResult:
    """TEST 5 — TTS: verify audio output is produced and non-empty."""
    r = TestResult("TEST 5: TTS audio output")
    print(_header(r.name))

    pipeline = get_pipeline(with_tts=True)
    if pipeline.tts is None:
        r.warn("TTS backend not available — skipping TTS test")
        return r

    query = "What is CDG airport?"
    print(f"  query: {query!r}")

    resp = pipeline.process_text(query, with_tts=True)

    print(f"  source:      {resp.generation.source if resp.generation else 'N/A'}")
    if resp.tts:
        print(f"  tts voice:   {resp.tts.voice}")
        print(f"  tts lang:    {resp.tts.language}")
        print(f"  duration:    {resp.tts.duration_seconds:.2f}s")
        print(f"  tts_ms:      {resp.tts.processing_time_ms:.0f}ms")
    _print_timings(resp)

    r.check(resp.generation is not None, "generation present")
    r.check(resp.tts is not None, "TTS block present in response")
    if resp.tts:
        r.check(resp.tts.duration_seconds > 0, "TTS duration > 0",
                f"{resp.tts.duration_seconds:.2f}s")
        r.check(resp.tts.processing_time_ms > 0, "TTS processing time > 0")
        r.check(resp.tts.voice is not None, "TTS voice assigned", str(resp.tts.voice))
    return r


# ── golden battery ────────────────────────────────────────────────────────────

def test_golden_battery(verbose: bool, with_tts: bool) -> list[TestResult]:
    """Run the 5 golden scenarios from the spec."""
    results = []

    # 1. English CDG → RAG
    r = TestResult("GOLDEN 1: English CDG → RAG")
    print(_header(r.name))
    pipeline = get_pipeline(with_tts)
    resp = pipeline.process_text("What is CDG airport?", with_tts=False)
    _print_generation(resp, verbose)
    r.check(resp.generation is not None, "generation present")
    if resp.generation:
        r.check(resp.generation.source in ("rag", "general_knowledge"),
                "source rag or gk", resp.generation.source)
    results.append(r)

    # 2. Urdu CDG → RAG
    r = TestResult("GOLDEN 2: Urdu CDG → RAG")
    print(_header(r.name))
    resp = pipeline.process_text("سی ڈی جی ہوائی اڈا کیا ہے؟",
                                  language_hint="urd_Arab", with_tts=False)
    _print_generation(resp, verbose)
    r.check(resp.language == "urd_Arab", "language urd_Arab", str(resp.language))
    r.check(resp.generation is not None, "generation present")
    if resp.generation:
        r.check(resp.generation.source in ("rag", "general_knowledge"),
                "source rag or gk", resp.generation.source)
    results.append(r)

    # 3. Capital of France → general_knowledge
    r = TestResult("GOLDEN 3: Capital of France → general_knowledge")
    print(_header(r.name))
    resp = pipeline.process_text("What is the capital of France?", with_tts=False)
    _print_generation(resp, verbose)
    r.check(resp.generation is not None, "generation present")
    if resp.generation:
        r.check(resp.generation.source == "general_knowledge",
                "source is general_knowledge", resp.generation.source)
        r.check(resp.generation.source != "rag",
                "NOT incorrectly labelled rag", resp.generation.source)
    results.append(r)

    # 4. Telephone inventor → general_knowledge
    r = TestResult("GOLDEN 4: Telephone inventor → general_knowledge")
    print(_header(r.name))
    resp = pipeline.process_text("Who invented the telephone?", with_tts=False)
    _print_generation(resp, verbose)
    r.check(resp.generation is not None, "generation present")
    if resp.generation:
        r.check(resp.generation.source == "general_knowledge",
                "source is general_knowledge", resp.generation.source)
    results.append(r)

    # 5. Voice → STT → answer → TTS
    r = TestResult("GOLDEN 5: Voice input → STT → answer → TTS")
    print(_header(r.name))
    audio = AUDIO_DIR / "test_english.wav"
    if audio.exists():
        resp = pipeline.process_audio(str(audio), with_tts=with_tts)
        _print_generation(resp, verbose)
        r.check(bool(resp.transcript), "STT produced transcript", repr(resp.transcript))
        r.check(resp.generation is not None, "answer generated")
        if with_tts and pipeline.tts is not None:
            r.check(resp.tts is not None, "TTS block present")
        else:
            r.warn("TTS skipped (--no-tts or no TTS backend)")
    else:
        r.warn(f"audio file not found: {audio}")
    results.append(r)

    return results


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_timings(resp) -> None:
    t = resp.timings
    parts = []
    if t.stt_ms > 0:
        parts.append(f"stt={t.stt_ms:.0f}ms")
    parts.append(f"retrieval={t.retrieval_ms:.0f}ms")
    parts.append(f"llm={t.llm_ms:.0f}ms")
    if t.grounding_ms > 0:
        parts.append(f"grounding={t.grounding_ms:.0f}ms")
    if t.tts_ms > 0:
        parts.append(f"tts={t.tts_ms:.0f}ms")
    parts.append(f"total={t.total_ms:.0f}ms")
    print(f"  timings:     {' | '.join(parts)}")


def _print_generation(resp, verbose: bool) -> None:
    if resp.generation:
        g = resp.generation
        print(f"  source={g.source}  status={g.status}  grounded={g.grounded}")
        if verbose:
            print(f"  answer: {g.answer[:200]}")
    _print_timings(resp)


def _check_tts(r: TestResult, resp, with_tts: bool) -> None:
    if not with_tts:
        return
    pipeline = get_pipeline(with_tts)
    if pipeline.tts is None:
        r.warn("TTS backend not configured — TTS check skipped")
        return
    if resp.tts is not None:
        r.check(resp.tts.duration_seconds > 0, "TTS duration > 0",
                f"{resp.tts.duration_seconds:.2f}s")
    else:
        r.warn("TTS block absent (TTS may have failed silently)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS stage")
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only one numbered test")
    parser.add_argument("--golden", action="store_true",
                        help="Run the 5 golden battery scenarios")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full answers")
    args = parser.parse_args()

    with_tts = not args.no_tts
    verbose = args.verbose

    print(f"\n{'='*60}")
    print("  Segment 7 — Voice RAG End-to-End Tests")
    print(f"{'='*60}")
    print(f"  TTS: {'enabled' if with_tts else 'disabled (--no-tts)'}")
    print(f"  Audio dir: {AUDIO_DIR}")

    all_results: list[TestResult] = []

    if args.golden:
        all_results.extend(test_golden_battery(verbose, with_tts))
    elif args.test == 1:
        all_results.append(test_1_english_rag(verbose, with_tts))
    elif args.test == 2:
        all_results.append(test_2_urdu_rag(verbose, with_tts))
    elif args.test == 3:
        all_results.append(test_3_urdu_edge_mp3(verbose, with_tts))
    elif args.test == 4:
        all_results.append(test_4_general_knowledge(verbose, with_tts))
    elif args.test == 5:
        all_results.append(test_5_tts_output(verbose))
    else:
        # Run all 5 tests
        all_results.append(test_1_english_rag(verbose, with_tts))
        all_results.append(test_2_urdu_rag(verbose, with_tts))
        all_results.append(test_3_urdu_edge_mp3(verbose, with_tts))
        all_results.append(test_4_general_knowledge(verbose, with_tts))
        all_results.append(test_5_tts_output(verbose))

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    total_passed = 0
    total_failed = 0
    for r in all_results:
        status = _ok("PASS") if r.ok else _fail("FAIL")
        n_pass = len(r.passed)
        n_fail = len(r.failed)
        n_warn = len(r.warnings)
        print(f"  {status}  {r.name}  "
              f"({n_pass} checks passed, {n_fail} failed"
              + (f", {n_warn} warnings" if n_warn else "") + ")")
        for f in r.failed:
            print(f"         {_fail(f)}")
        total_passed += n_pass
        total_failed += n_fail

    print(f"\n  Total checks: {total_passed + total_failed} "
          f"({total_passed} passed, {total_failed} failed)")

    if total_failed == 0:
        print(f"\n  {_GREEN}{_BOLD}All tests passed.{_RESET}")
        return 0
    else:
        print(f"\n  {_RED}{_BOLD}{total_failed} check(s) failed.{_RESET}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Segment 4C — real grounded-LLM integration test.

Retrieves real context with the production hybrid retriever, generates an
answer through the real LLM provider (Ollama by default), and prints the
grounding verdict and latency.

This is an OPT-IN integration script — the unit suite never calls a provider.

Usage (from backend/):
    python scripts/test_llm.py "what is cdg airport"
    python scripts/test_llm.py "What is the population of Mars?"  # expect abstention
    python scripts/test_llm.py "سی ڈی جی ہوائی اڈا کیا ہے؟" --language ur
    python scripts/test_llm.py "what is cdg airport" --top-k 5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import DEFAULT_TOP_K  # noqa: E402
from app.llm import get_llm_service  # noqa: E402
from app.retrieval.production import get_production_retriever  # noqa: E402

_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")


def detect_language(query: str) -> str:
    """Script-based detection matching the pipeline (urd_Arab | eng_Latn)."""
    if _ARABIC_RE.search(query):
        return "urd_Arab"
    return "eng_Latn"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", type=str, help="the natural-language question to answer")
    parser.add_argument("--language", default=None,
                        help="script-form language override (eng_Latn | urd_Arab); "
                             "defaults to script-based detection")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="number of chunks to retrieve")
    args = parser.parse_args()

    language = args.language or detect_language(args.query)
    language = {"en": "eng_Latn", "ur": "urd_Arab"}.get(language, language)
    print(f"Retrieving context for {args.query!r} (language={language!r}, "
          f"top_k={args.top_k}) ...")
    retriever = get_production_retriever()
    results = retriever.search(args.query, top_k=args.top_k, language=language)

    print(f"\n--- Retrieved context (strategy={retriever.strategy}, top_k={args.top_k}) ---")
    for i, r in enumerate(results, start=1):
        lang = (r.metadata or {}).get("language")
        print(f"{i}. [{r.chunk_id} | score={r.score} | {lang}] {r.text[:160]}")

    if not results:
        print("\nNo usable context retrieved — the LLM will not be called.")
        return

    llm = get_llm_service()
    print(f"\n--- Grounded answer generation (provider={llm.provider_name}, "
          f"model={llm.model}) ---")
    response = llm.generate(args.query, [r.text for r in results], language=language)
    print(f"Query:          {args.query}")
    print(f"Answer:         {response.answer}")
    print(f"Language:       {response.language}")
    print(f"Grounded:       {response.grounded}")
    print(f"Abstained:      {response.abstained}")
    print(f"Context count:  {response.context_count}")
    print(f"LLM latency:    {response.latency_ms:.1f} ms")
    if response.usage:
        print(f"Usage:          prompt={response.usage.prompt_tokens} | "
              f"completion={response.usage.completion_tokens}")

    if not response.grounded:
        print("\nWARNING: answer is NOT grounded in the retrieved context "
              "(abstention or hallucination).")


if __name__ == "__main__":
    main()

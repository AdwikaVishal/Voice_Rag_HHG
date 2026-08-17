"""Segment 4 — validate the production retriever on the winning strategy.

Loads the production hybrid retriever (FAISS + BM25 + RRF, default strategy
``recursive`` from ``app/config.py``) from the existing index and verifies it
end to end:

* the index loads and reports its chunk count,
* English and Urdu queries both return ranked results,
* every result carries the metadata downstream stages need (chunk_id,
  record_id, query_id, passage_index, language, text, prev/next links).

No embedding / indexing is performed; the existing benchmark index is reused.

Usage (from backend/):
    python scripts/validate_production_retriever.py [--strategy recursive]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import CHUNKING_STRATEGY  # noqa: E402
from app.retrieval import ProductionRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO)

REQUIRED_METADATA_FIELDS = (
    "chunk_id",
    "record_id",
    "query_id",
    "passage_index",
    "language",
    "text",
    "prev_chunk_id",
    "next_chunk_id",
)

SAMPLE_QUERIES = (
    ("eng_Latn", "what is a corporation?"),
    ("urd_Arab", "کارپوریشن کیا ہے؟"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default=CHUNKING_STRATEGY)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = ProductionRetriever(strategy=args.strategy)
    print(f"Strategy:        {retriever.strategy}")
    print(f"Index directory: {retriever.index_dir}")

    failures = 0
    for expected_lang, query in SAMPLE_QUERIES:
        results = retriever.search(query, top_k=args.top_k)
        print(f"\nQuery [{expected_lang}]: {query!r}")
        print(f"  -> {len(results)} results (expected >= 1)")
        if not results:
            failures += 1
            continue
        for r in results:
            missing = [f for f in REQUIRED_METADATA_FIELDS if f not in r.metadata]
            lang = r.metadata.get("language")
            if missing or lang != expected_lang:
                failures += 1
            print(
                f"    #{r.rank} {r.chunk_id}  score={r.score:.4f}  lang={lang} "
                f"passage={r.metadata.get('passage_index')} "
                f"next={r.metadata.get('next_chunk_id')}  | {r.text[:60]!r}"
            )

    print(f"\nChunk count in index: {retriever.chunk_count}")
    print(f"Validation {'PASSED' if failures == 0 else f'FAILED ({failures} problems)'}")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()

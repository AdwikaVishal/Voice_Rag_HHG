"""Segment 7B — BEFORE (benchmark sample) vs AFTER (full corpus) retrieval check.

Runs three queries whose selected-answer passages exist ONLY in the full
5,000-record corpus (not in the 495-record benchmark sample) through the hybrid
retriever against each index and reports the rank of the first relevant
(is_selected == 1) chunk.

No writes; diagnosis only.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.retrieval import EmbeddingService, FAISSStore, HybridRetriever  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402

QUERIES = [
    ("rec-000001", "what is a corporation?", "msmarco_xi_000001"),
    ("rec-000028", "what is barter system and its problems", "msmarco_xi_000028"),
    ("rec-000042", "sty causes", "msmarco_xi_000042"),
]

BENCH = BASE_DIR / "data" / "indexes" / "benchmark" / "recursive"
FULL = BASE_DIR / "data" / "indexes" / "recursive"


def first_relevant(results) -> tuple[int | None, str | None]:
    for r in results:
        if r.metadata.get("is_selected") == 1:
            return r.rank, r.chunk_id
    return None, None


def main() -> None:
    embeddings = EmbeddingService(show_progress=False)

    for label, path in (("BENCHMARK (BEFORE)", BENCH), ("FULL (AFTER)", FULL)):
        dense = FAISSStore.load(path)
        bm25 = BM25Retriever.load(path)
        hybrid = HybridRetriever(dense=dense, bm25=bm25, embeddings=embeddings)
        print(f"\n=== {label}  index={path}  chunks={dense.index.ntotal} ===")
        for name, query, rec_id in QUERIES:
            results = hybrid.search(query, top_k=10)
            rank, cid = first_relevant(results)
            rel = f"rank={rank} chunk={cid}" if rank else "NOT RETRIEVED in top-10"
            print(f"  {name:<14} {query!r}  record={rec_id}  -> {rel}")
            print(f"     top-3: {[r.metadata.get('record_id') + '::' + str(r.metadata.get('passage_index')) for r in results[:3]]}")


if __name__ == "__main__":
    main()

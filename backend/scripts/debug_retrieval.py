"""Segment 7A — retrieval quality audit (diagnosis ONLY; no behavior changes).

Loads the production index (recursive benchmark sample, ~9,964 chunks) and the
same components the API uses (EmbeddingService + FAISSStore + BM25Retriever +
HybridRetriever), then runs a small query battery through each retriever
separately so the failure point can be isolated:

    A. BM25 only
    B. Dense / FAISS only
    C. Hybrid (dense + BM25 + RRF)

Also reports:
    * corpus / index stats (records, chunks, languages, sample vs full)
    * lexical coverage check per query against the INDEXED chunks only
    * score distribution per query

No index is built, no embeddings are recomputed and no file is written.

Usage (from backend/):
    python scripts/debug_retrieval.py
    python scripts/debug_retrieval.py --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import CHUNKING_STRATEGY, strategy_index_dir  # noqa: E402
from app.retrieval import EmbeddingService, FAISSStore, HybridRetriever  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402
from app.retrieval.filters import detect_script_language  # noqa: E402
from app.retrieval.production import ProductionRetriever  # noqa: E402

EXCERPT = 110

QUERIES: list[tuple[str, str]] = [
    ("Q1", "What is the capital of France?"),
    ("Q2", "Which airport serves Paris and is abbreviated as CDG?"),
    ("Q3", "Who invented the telephone?"),
    ("Q4", "What is CDG airport?"),
    ("Q5", "سی ڈی جی ہوائی اڈا کیا ہے؟"),
]

# (label, list of lexical evidence markers) — searched ONLY in the indexed
# chunks (i.e. the corpus the API actually retrieves from).
COVERAGE: dict[str, list[str]] = {
    "Q1": ["capital of france", "paris is the capital", "capital"],
    "Q2": ["cdg", "roissy charles de gaulle", "which airport is cdg"],
    "Q3": ["invented", "telephone", "graham bell", "alexander bell"],
    "Q4": ["cdg", "roissy charles de gaulle", "which airport is cdg"],
    "Q5": ["سی ڈی جی", "ہوائی اڈا", "پیرس"],
}

SAMPLE_RECORDS_PATH = BASE_DIR / "data" / "processed" / "benchmark_manifest.json"


def excerpt(text: str, n: int = EXCERPT) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def print_result_row(r, label: str) -> None:
    m = r.metadata
    print(
        f"  #{r.rank:<2} score={r.score:.4f} chunk={r.chunk_id} "
        f"rec={m.get('record_id')} qid={m.get('query_id')} "
        f"sel={m.get('is_selected')} lang={m.get('language')} "
        f"passage={m.get('passage_index')} | {excerpt(r.text)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--strategy", default=CHUNKING_STRATEGY)
    args = parser.parse_args()

    index_dir = strategy_index_dir(args.strategy)
    if not (index_dir / "index.faiss").exists():
        print(f"ERROR: index not found at {index_dir} — run build_retrieval_index.py first")
        sys.exit(1)

    print("=" * 78)
    print("SEGMENT 7A — RETRIEVAL QUALITY AUDIT (diagnosis only, no fixes)")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Components (same as production: production.py uses these exact classes)
    # ------------------------------------------------------------------
    prod = ProductionRetriever(strategy=args.strategy)
    prod._ensure_loaded()
    dense = prod._dense
    bm25 = prod._bm25
    embeddings = prod._embeddings
    hybrid = prod._hybrid

    print(f"\nStrategy:      {prod.strategy}")
    print(f"Index dir:     {index_dir}")
    print(f"Index chunks:  {dense.index.ntotal}")
    print(f"Embedding:     {embeddings.model_name} (dim={embeddings.dimension})")
    print(f"FAISS index:   {dense.index.__class__.__name__}")
    print(f"Hybrid RRF k:  {hybrid.rrf_k}, expansion: {hybrid.expansion}")
    print(f"Language flag: {index_dir / 'bm25.pkl'} exists -> BM25 loaded from index")

    # ------------------------------------------------------------------
    # Corpus stats (index = what the API retrieves from)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("INDEXED CORPUS STATS")
    print("-" * 78)
    lang_count: dict[str, int] = {}
    records: set[str] = set()
    for c in dense.chunks:
        lang_count[c.get("language", "?")] = lang_count.get(c.get("language", "?"), 0) + 1
        records.add(c.get("record_id", "?"))
    print(f"  indexed chunks : {len(dense.chunks)}")
    print(f"  indexed records: {len(records)}")
    for lang, n in sorted(lang_count.items()):
        print(f"  {lang:<12}: {n}")
    if SAMPLE_RECORDS_PATH.exists():
        manifest = json.loads(SAMPLE_RECORDS_PATH.read_text(encoding="utf-8"))
        print(
            f"  source sample  : {manifest.get('records_sampled')}/{manifest.get('total_records')} "
            f"records (seed={manifest.get('seed')}, target={manifest.get('target_sample_size')} chunks) "
            f"— this is a SUBSET, not the full MSMARCO-XI corpus"
        )
    else:
        print("  (manifest missing — cannot confirm sample provenance)")

    # ------------------------------------------------------------------
    # Coverage check (evidence exists IN THE INDEXED CORPUS?)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("COVERAGE CHECK — is evidence in the INDEXED corpus?")
    print("-" * 78)
    chunk_texts = {c["chunk_id"]: c for c in dense.chunks}
    for label, query in QUERIES:
        print(f"\n  {label} {query!r}")
        for marker in COVERAGE[label]:
            hits = [cid for cid, c in chunk_texts.items() if marker in c.get("text", "").lower()]
            if hits:
                print(f"    FOUND  marker={marker!r:28} -> {len(hits)} chunk(s); first: {hits[0]}")
            else:
                print(f"    absent marker={marker!r:28}")

    # ------------------------------------------------------------------
    # Per-query diagnostics: BM25 / dense / hybrid
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"PER-QUERY DIAGNOSTICS (top-{args.top_k})")
    print("-" * 78)

    for label, query in QUERIES:
        print(f"\n{'=' * 78}\n{label}: {query!r}")
        print(f"  detected script language: {detect_script_language(query)!r}")
        qvec = embeddings.embed_query(query)

        bm25_results = bm25.search(query, top_k=args.top_k)
        dense_results = dense.search(qvec, top_k=args.top_k)
        hybrid_results = hybrid.search_embedded(qvec, query, top_k=args.top_k)

        print(f"\n  A. BM25 only ({len(bm25_results)} results)")
        for r in bm25_results:
            print_result_row(r, "BM25")

        print(f"\n  B. Dense only ({len(dense_results)} results)")
        for r in dense_results:
            print_result_row(r, "dense")

        print(f"\n  C. Hybrid ({len(hybrid_results)} results)")
        for r in hybrid_results:
            print_result_row(r, "hybrid")

    # ------------------------------------------------------------------
    # Score distribution
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("SCORE DISTRIBUTION (per mode, top-1 / top-5 range / gap 1->5)")
    print("-" * 78)
    for label, query in QUERIES:
        qvec = embeddings.embed_query(query)
        bm25_results = bm25.search(query, top_k=5)
        dense_results = dense.search(qvec, top_k=5)
        hybrid_results = hybrid.search_embedded(qvec, query, top_k=5)
        print(f"\n  {label}: {query!r}")
        for mode, results in (("BM25", bm25_results), ("dense", dense_results), ("hybrid", hybrid_results)):
            if not results:
                print(f"    {mode:<6}: no results")
                continue
            s = [r.score for r in results]
            top5 = f"{min(s):.4f}" if len(s) == 5 else f"{min(s):.4f} (n={len(s)})"
            gap = s[0] - s[-1] if len(s) > 1 else 0.0
            print(
                f"    {mode:<6}: top1={s[0]:.4f}  top5_min={top5}  "
                f"gap1->{len(s)}={gap:.4f}"
            )


if __name__ == "__main__":
    main()

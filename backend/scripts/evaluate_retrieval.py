"""Segment 3 — retrieval evaluation across chunking strategies.

Measures Recall@1/3/5/10 for dense (FAISS), sparse (BM25) and hybrid
(FAISS + BM25 + RRF) retrieval for every chunking strategy, plus P50/P70/P100
per-stage latency. Uses the benchmark sample manifest written by
``scripts/build_retrieval_index.py`` so every strategy is scored on the same
query set and the same underlying source records.

Ground truth: a chunk is relevant to a query iff its ``(record_id,
passage_index)`` is a selected passage (``is_selected == 1``) of that query.
Relevance is language-agnostic — the English and Urdu chunks of a selected
passage both count. No labels are fabricated.

``--audit`` mode (Segment 3 extension) runs the *production* index (default
``data/indexes/recursive``) against a hand-authored query battery — known
answerable (CDG English + Urdu), missing-evidence, and noise queries — and
prints one row per query::

    Query | Language | Expected Evidence | BM25 Rank | Dense Rank | Hybrid Rank | Status

Ranks come from actually executing each retriever (BM25, dense, hybrid-RRF)
independently; nothing is hardcoded. Queries whose expected evidence is not in
the indexed corpus are reported as ``NO-EVIDENCE`` (a coverage failure, not a
retrieval failure) and are never forced to pass.

Two explicit evaluation modes select the corpus, manifest and index, and print
them (``Mode`` / ``Manifest`` / ``Index`` / ``Records`` / ``Chunks``) before the
embedding model is loaded — there is no silent fallback to a different index:

* ``--mode benchmark`` (default) — the reproducible 495-record sample manifest
  (``data/processed/benchmark_manifest.json``) scored across every chunking
  strategy in ``data/indexes/benchmark/{strategy}``.
* ``--mode full`` — all 5,000 records of ``data/processed/msmarco_xi_sample.jsonl``
  scored against the production index ``data/indexes/recursive`` (the index the
  API serves).

``--smoke`` runs a fixed 5-query battery (CDG English + Urdu, plus two queries
whose evidence is absent) against the mode's index and exits, printing each
query's language, BM25/dense/hybrid top result, latency, and an explicit
``EVIDENCE_EXISTS=YES/NO`` → ``RETRIEVAL=PASS/FAIL/NOT_APPLICABLE`` verdict.
The embedding model is loaded exactly once and reused for all queries.

Usage (from backend/):
    python scripts/evaluate_retrieval.py                        # benchmark mode (default)
    python scripts/evaluate_retrieval.py --mode full            # 5,000-record production index
    python scripts/evaluate_retrieval.py --strategy recursive   # benchmark, one strategy
    python scripts/evaluate_retrieval.py --eval-queries 200 --rebuild-eval
    python scripts/evaluate_retrieval.py --smoke                # 5-query smoke battery
    python scripts/evaluate_retrieval.py --mode full --smoke    # smoke against production index
    python scripts/evaluate_retrieval.py --audit                # production-index audit table
    python scripts/evaluate_retrieval.py --audit --index-dir data/indexes/recursive
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Never use TensorFlow/JAX: older sentence-transformers (via transformers
# integration hooks) can lazily import TensorFlow while the embedding model is
# loading, and TF's shared-library preload deadlocks on macOS ("[mutex.cc : 452]
# RAW: Lock blocking ..."). Opt out before anything can import either.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.chunking.factory import available_strategies  # noqa: E402
from app.retrieval import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    EmbeddingService,
    FAISSStore,
    HybridRetriever,
    aggregate_recall,
    build_eval_queries,
    chunks_for_records,
    detect_script_language,
    load_eval_set,
    load_chunk_records,
    percentiles,
    recall_at_k,
    relevant_chunk_ids,
    save_eval_set,
)
from app.retrieval.bm25 import BM25Retriever  # noqa: E402
from app.retrieval.embeddings import DEFAULT_DIMENSION  # noqa: E402
from app.retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402

# Imported after app.retrieval so torch (loaded by app.retrieval) always
# precedes faiss-cpu — the safe import order on macOS + Python 3.13.
import faiss  # noqa: E402

logger = logging.getLogger("evaluate_retrieval")

DEFAULT_CORPUS = BASE_DIR / "data" / "processed" / "msmarco_xi_sample.jsonl"
DEFAULT_CHUNKS_DIR = BASE_DIR / "data" / "processed" / "chunks"
DEFAULT_INDEXES_DIR = BASE_DIR / "data" / "indexes" / "benchmark"
DEFAULT_MANIFEST = BASE_DIR / "data" / "processed" / "benchmark_manifest.json"
DEFAULT_EVAL_PATH = BASE_DIR / "data" / "processed" / "retrieval_eval.jsonl"
DEFAULT_RESULTS = BASE_DIR / "data" / "processed" / "retrieval_results.json"
DEFAULT_AUDIT_PATH = BASE_DIR / "data" / "processed" / "retrieval_audit.json"
DEFAULT_EVAL_QUERIES = 500

# Full (production) mode: all 5,000 records scored against the API's index.
FULL_INDEXES_DIR = BASE_DIR / "data" / "indexes" / "recursive"
FULL_EVAL_PATH = BASE_DIR / "data" / "processed" / "retrieval_eval_full.jsonl"
FULL_RESULTS = BASE_DIR / "data" / "processed" / "retrieval_results_full.json"

RECALL_KS = (1, 3, 5, 10)
TOP_K = 10

# Production-index audit battery. ``evidence`` is an optional ``(record_id,
# passage_index)`` pair naming the chunk that answers the query; ``None`` means
# no such evidence is expected to exist in the indexed corpus.
AUDIT_QUERIES: list[dict] = [
    {
        "label": "A1",
        "query": "What is CDG airport?",
        "language": "eng_Latn",
        "evidence": ("msmarco_xi_000917", 3),
    },
    {
        "label": "A2",
        "query": "Which airport serves Paris and is abbreviated as CDG?",
        "language": "eng_Latn",
        "evidence": ("msmarco_xi_000917", 3),
    },
    {
        "label": "A3",
        "query": "سی ڈی جی ہوائی اڈا کیا ہے؟",
        "language": "urd_Arab",
        "evidence": ("msmarco_xi_000917", 3),
    },
    {
        "label": "D1",
        "query": "What is the capital of France?",
        "language": "eng_Latn",
        "evidence": None,
    },
    {
        "label": "D2",
        "query": "Who invented the telephone?",
        "language": "eng_Latn",
        "evidence": None,
    },
    {
        "label": "E1",
        "query": "What is the weather in Tokyo today?",
        "language": "eng_Latn",
        "evidence": None,
    },
    {
        "label": "E2",
        "query": "Recipe for chocolate chip cookies?",
        "language": "eng_Latn",
        "evidence": None,
    },
    {
        "label": "E3",
        "query": "How do I fix a leaking faucet?",
        "language": "eng_Latn",
        "evidence": None,
    },
    {
        "label": "E4",
        "query": "کیا آپ مجھے ایک مزیدار کیک بنانے کا طریقہ بتا سکتے ہیں؟",
        "language": "urd_Arab",
        "evidence": None,
    },
]

# Fixed 5-query smoke battery. ``evidence`` is an optional ``(record_id,
# passage_index)`` naming the indexed chunk that answers the query; ``None``
# means no such evidence exists in the corpus (S4/S5 are expected to be
# reported NOT_APPLICABLE — a coverage gap, not a retrieval failure).
SMOKE_QUERIES: list[dict] = [
    {
        "label": "S1",
        "query": "What is CDG airport?",
        "language": "eng_Latn",
        "evidence": ("msmarco_xi_000917", 3),
    },
    {
        "label": "S2",
        "query": "Which airport serves Paris and is abbreviated as CDG?",
        "language": "eng_Latn",
        "evidence": ("msmarco_xi_000917", 3),
    },
    {
        "label": "S3",
        "query": "سی ڈی جی ہوائی اڈا کیا ہے؟",
        "language": "urd_Arab",
        "evidence": ("msmarco_xi_000917", 3),
    },
    {
        "label": "S4",
        "query": "What is the capital of France?",
        "language": "eng_Latn",
        "evidence": None,
    },
    {
        "label": "S5",
        "query": "Who invented the telephone?",
        "language": "eng_Latn",
        "evidence": None,
    },
]


class Timings:
    """Millisecond latency buckets for per-stage retrieval timing."""

    def __init__(self) -> None:
        self.buckets: dict[str, list[float]] = {}

    def record(self, bucket: str, seconds: float) -> None:
        self.buckets.setdefault(bucket, []).append(seconds * 1000.0)

    def raw(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self.buckets.items()}

    def summary(self) -> dict[str, dict]:
        out = {}
        for bucket, values in self.buckets.items():
            out[bucket] = percentiles(values)
            out[bucket]["mean_ms"] = round(statistics.mean(values), 3)
            out[bucket]["median_ms"] = round(statistics.median(values), 3)
            out[bucket]["count"] = len(values)
        return out


def evaluate_strategy(
    strategy: str,
    queries: list[dict],
    dense: FAISSStore,
    bm25: BM25Retriever,
    hybrid: HybridRetriever,
    embeddings: EmbeddingService,
    relevant: dict,
) -> dict:
    timings = Timings()
    dense_hits: list[list[float]] = []
    bm25_hits: list[list[float]] = []
    hybrid_hits: list[list[float]] = []
    queries_evaluated = 0
    queries_without_relevant = 0
    queries_language_known = 0

    # Warm up the model + retrievers so the first strategy is not penalized by
    # torch thread-pool / FAISS / BM25 lazy initialization.
    for _warm in range(3):
        embeddings.embed_query("warmup query کارپوریشن")
        dense.search(np.ones(embeddings.dimension, dtype=np.float32), top_k=TOP_K)
        bm25.search("warmup query کارپوریشن", top_k=TOP_K)
        hybrid.search_embedded(
            np.ones(embeddings.dimension, dtype=np.float32), "warmup query کارپوریشن", top_k=TOP_K
        )

    for query in queries:
        qid = query["query_id"]
        relevant_ids = set(relevant.get(qid, set()))
        if not relevant_ids:
            queries_without_relevant += 1
            continue
        queries_evaluated += 1
        if hybrid._resolve_language(query["query"], None):
            queries_language_known += 1

        embed_start = time.perf_counter()
        qvec = embeddings.embed_query(query["query"])
        timings.record("query_embedding_ms", time.perf_counter() - embed_start)

        t0 = time.perf_counter()
        dense_results = dense.search(qvec, top_k=TOP_K)
        timings.record("faiss_ms", time.perf_counter() - t0)

        t0 = time.perf_counter()
        bm25_results = bm25.search(query["query"], top_k=TOP_K)
        timings.record("bm25_ms", time.perf_counter() - t0)

        t0 = time.perf_counter()
        hybrid_results = hybrid.search_embedded(qvec, query["query"], top_k=TOP_K)
        hybrid_end = time.perf_counter()
        timings.record("hybrid_search_ms", hybrid_end - t0)

        t0 = time.perf_counter()
        reciprocal_rank_fusion([dense_results, bm25_results], k=hybrid.rrf_k)
        timings.record("rrf_ms", time.perf_counter() - t0)

        timings.record("total_retrieval_ms", hybrid_end - embed_start)

        dense_ids = [r.chunk_id for r in dense_results]
        bm25_ids = [r.chunk_id for r in bm25_results]
        hybrid_ids = [r.chunk_id for r in hybrid_results]

        dense_hits.append([recall_at_k(dense_ids, relevant_ids, k) for k in RECALL_KS])
        bm25_hits.append([recall_at_k(bm25_ids, relevant_ids, k) for k in RECALL_KS])
        hybrid_hits.append([recall_at_k(hybrid_ids, relevant_ids, k) for k in RECALL_KS])

    return {
        "queries_evaluated": queries_evaluated,
        "queries_without_relevant_chunks": queries_without_relevant,
        "queries_language_known": queries_language_known,
        "dense": aggregate_recall(dense_hits, RECALL_KS),
        "bm25": aggregate_recall(bm25_hits, RECALL_KS),
        "hybrid": aggregate_recall(hybrid_hits, RECALL_KS),
        "latency_ms": timings.summary(),
        "latency_raw_ms": timings.raw(),
    }


def latency_table(strategy_stats: dict) -> dict:
    """Per-mode P50/P70/P100 latency (embedding included where applicable)."""
    lat = strategy_stats["latency_raw_ms"]
    dense_values = [a + b for a, b in zip(lat["query_embedding_ms"], lat["faiss_ms"])]
    hybrid_values = lat["total_retrieval_ms"]
    bm25_values = lat["bm25_ms"]
    return {
        "dense": percentiles(dense_values),
        "bm25": percentiles(bm25_values),
        "hybrid": percentiles(hybrid_values),
    }


# --------------------------------------------------------------------------
# Production-index audit (per-query BM25 / dense / hybrid rank table)
# --------------------------------------------------------------------------

def _evidence_chunk_ids(chunks: list[dict], record_id: str, passage_index: int) -> list[str]:
    """Chunk_ids in the indexed corpus covering one ``(record_id, passage_index)``."""
    return [
        c["chunk_id"]
        for c in chunks
        if c.get("record_id") == record_id and c.get("passage_index") == passage_index
    ]


def _rank_of(ids: list[str], evidence_ids: set[str]) -> int | None:
    """1-based rank of the first evidence chunk in ``ids`` (None if absent)."""
    for i, cid in enumerate(ids):
        if cid in evidence_ids:
            return i + 1
    return None


def run_audit(
    index_dir: Path,
    model: str,
    batch_size: int,
    top_k: int,
    embeddings: EmbeddingService | None = None,
) -> dict:
    """Run the hand-authored query battery against one index.

    For every query the dense, BM25 and hybrid retrievers are executed
    independently (nothing is hardcoded) and each mode's rank of the expected
    evidence chunk is reported. Queries whose expected evidence is not in the
    indexed corpus are reported as ``NO-EVIDENCE``.
    """
    dense = FAISSStore.load(index_dir)
    bm25 = BM25Retriever.load(index_dir)
    if embeddings is None:
        embeddings = EmbeddingService(model_name=model, batch_size=batch_size, show_progress=False)
    hybrid = HybridRetriever(dense=dense, bm25=bm25, embeddings=embeddings)

    # Warm up the model + retrievers (lazy init should not skew results).
    for _ in range(2):
        embeddings.embed_query("warmup query کارپوریشن")
        dense.search(np.ones(embeddings.dimension, dtype=np.float32), top_k=top_k)
        bm25.search("warmup query کارپوریشن", top_k=top_k)

    rows: list[dict] = []
    for entry in AUDIT_QUERIES:
        query = entry["query"]
        language = entry["language"]
        evidence_ids: set[str] = set()
        evidence_present = False
        if entry["evidence"] is not None:
            evidence_ids = set(
                _evidence_chunk_ids(dense.chunks, *entry["evidence"])
            )
            evidence_present = bool(evidence_ids)

        qvec = embeddings.embed_query(query)
        dense_results = dense.search(qvec, top_k=top_k, language=language)
        bm25_results = bm25.search(query, top_k=top_k, language=language)
        hybrid_results = hybrid.search_embedded(qvec, query, top_k=top_k, language=language)

        bm25_rank: int | None = None
        dense_rank: int | None = None
        hybrid_rank: int | None = None
        if not evidence_present:
            status = "NO-EVIDENCE"
        else:
            bm25_rank = _rank_of([r.chunk_id for r in bm25_results], evidence_ids)
            dense_rank = _rank_of([r.chunk_id for r in dense_results], evidence_ids)
            hybrid_rank = _rank_of([r.chunk_id for r in hybrid_results], evidence_ids)
            status = "PASS" if hybrid_rank is not None else "FAIL"

        row = {
            "label": entry["label"],
            "query": query,
            "language": entry["language"],
            "detected_language": detect_script_language(query),
            "evidence_record_passage": entry["evidence"],
            "evidence_chunk_ids": sorted(evidence_ids) if evidence_ids else None,
            "evidence_present": evidence_present,
            "bm25_rank": bm25_rank,
            "dense_rank": dense_rank,
            "hybrid_rank": hybrid_rank,
            "status": status,
            "bm25_top1": bm25_results[0].score if bm25_results else None,
            "dense_top1": dense_results[0].score if dense_results else None,
            "hybrid_top1": hybrid_results[0].score if hybrid_results else None,
            "hybrid_top1_text": (hybrid_results[0].text[:120] if hybrid_results else ""),
        }
        rows.append(row)
    return {
        "index_dir": str(index_dir),
        "chunks": dense.index.ntotal,
        "model": model,
        "top_k": top_k,
        "rows": rows,
    }


def print_audit_table(report: dict) -> None:
    header = (
        f"{'Query':<72} {'Lang':<9} {'Evid':<4} {'BM25':>5} {'Dense':>6} {'Hybrid':>7}  Status"
    )
    print("")
    print(header)
    print("-" * len(header))
    for row in report["rows"]:
        query = row["query"] if len(row["query"]) <= 46 else row["query"][:45] + "…"
        evid = "yes" if row["evidence_present"] else "no"
        bm25 = str(row["bm25_rank"]) if row["bm25_rank"] is not None else "—"
        dense = str(row["dense_rank"]) if row["dense_rank"] is not None else "—"
        hybrid = str(row["hybrid_rank"]) if row["hybrid_rank"] is not None else "—"
        print(
            f"{query:<72} {row['language']:<9} {evid:<4} {bm25:>5} {dense:>6} {hybrid:>7}  {row['status']}"
        )


# --------------------------------------------------------------------------
# Evaluation modes: benchmark (reproducible 495-record sample) vs full
# (5,000-record production index). The mode is resolved and validated BEFORE
# the embedding model is loaded; a wrong or missing index is a hard error.
# --------------------------------------------------------------------------

def _all_record_ids(corpus_path: Path) -> set[str]:
    """Every record_id present in the corpus file (full mode's manifest)."""
    record_ids: set[str] = set()
    with Path(corpus_path).open(encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                record_ids.add(json.loads(line).get("record_id"))
    return record_ids


def resolve_mode(args: argparse.Namespace) -> dict:
    """Resolve the corpus, manifest, indexes and record set for a mode."""
    if args.mode == "full":
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            logger.error("Full-mode corpus not found: %s", corpus_path)
            sys.exit(1)
        record_ids = _all_record_ids(corpus_path)
        index_dir = Path(args.index_dir) if args.index_dir else FULL_INDEXES_DIR
        return {
            "mode": "full",
            "manifest_path": corpus_path,
            "manifest": {"source": str(corpus_path), "total_records": len(record_ids)},
            "record_ids": record_ids,
            "records": len(record_ids),
            "indexes_dir": index_dir,
            "primary_index": index_dir,
            "strategies": ["recursive"],
        }

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(
            "Manifest not found: %s (run scripts/build_retrieval_index.py first)",
            manifest_path,
        )
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_ids = set(manifest["record_ids"])
    return {
        "mode": "benchmark",
        "manifest_path": manifest_path,
        "manifest": manifest,
        "record_ids": record_ids,
        "records": len(record_ids),
        "indexes_dir": Path(args.indexes_dir),
        "primary_index": Path(args.indexes_dir) / "recursive",
        "strategies": sorted(set(args.strategy or available_strategies())),
    }


def validate_index(index_dir: Path, mode: str, expected_records: int) -> dict:
    """Sanity-check one index before the model is loaded.

    Fails loudly (exit 1) on a missing, truncated or mode-mismatched index —
    never silently falls back to a different (smaller) index. Returns
    ``{chunks, records, languages, dim}`` for the mode header.
    """
    index_dir = Path(index_dir)
    index_file = index_dir / "index.faiss"
    meta_file = index_dir / "metadata.json"
    problems: list[str] = []
    if not index_file.exists():
        problems.append(f"missing {index_file}")
    if not meta_file.exists():
        problems.append(f"missing {meta_file}")
    if problems:
        for problem in problems:
            logger.error("Index validation failed: %s", problem)
        sys.exit(1)

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta_count = int(meta.get("count", -1))
    chunks = meta.get("chunks") or []
    if meta_count != len(chunks):
        problems.append(f"metadata count {meta_count} != metadata chunks {len(chunks)}")
    if int(meta.get("dim", -1)) != DEFAULT_DIMENSION:
        problems.append(
            f"embedding dim {meta.get('dim')} != expected {DEFAULT_DIMENSION}"
        )

    index = faiss.read_index(str(index_file))
    if int(index.ntotal) != meta_count:
        problems.append(f"FAISS ntotal {index.ntotal} != metadata count {meta_count}")
    if int(index.d) != DEFAULT_DIMENSION:
        problems.append(f"FAISS dim {index.d} != expected {DEFAULT_DIMENSION}")

    records = {c.get("record_id") for c in chunks}
    if len(records) != expected_records:
        problems.append(
            f"index covers {len(records)} records, expected {expected_records} "
            f"for --mode {mode}"
        )

    languages: dict[str, int] = {}
    for chunk in chunks:
        languages[chunk.get("language")] = languages.get(chunk.get("language"), 0) + 1
    unknown = {lang for lang in languages if lang not in {"eng_Latn", "urd_Arab"}}
    if unknown:
        problems.append(f"unexpected language labels in index: {sorted(unknown)}")

    if problems:
        for problem in problems:
            logger.error("Index validation failed (%s): %s", index_dir, problem)
        logger.error("Refusing to evaluate on an invalid or mode-mismatched index (no fallback).")
        sys.exit(1)

    logger.info("Validated index %s: %d chunks, %d records, %s",
                index_dir, meta_count, len(records), " ".join(f"{k}={v}" for k, v in sorted(languages.items())))
    return {
        "chunks": meta_count,
        "records": len(records),
        "languages": languages,
        "dim": int(meta.get("dim")),
    }


def print_mode_header(
    mode: str,
    manifest_path: Path,
    index_dir: Path,
    records: int,
    chunks: int,
    languages: dict[str, int],
) -> None:
    """Print the mode facts BEFORE the embedding model is loaded."""
    print("")
    print(f"Mode: {mode}")
    print(f"Manifest: {manifest_path}")
    print(f"Index: {index_dir}")
    print(f"Records: {records}")
    print(f"Chunks: {chunks}")
    print(f"Languages: " + " ".join(f"{k}={v}" for k, v in sorted(languages.items())))
    print("")


# --------------------------------------------------------------------------
# Fixed 5-query smoke battery (exactly SMOKE_QUERIES, one shared model)
# --------------------------------------------------------------------------

def run_smoke(
    index_dir: Path,
    model: str,
    batch_size: int,
    top_k: int,
    embeddings: EmbeddingService | None = None,
) -> dict:
    """Run the 5 fixed queries against one index.

    One EmbeddingService / FAISSStore / BM25Retriever / HybridRetriever is
    created and reused for all queries — no per-query service creation, no
    multiprocessing, sequential evaluation. ``model_loads`` counts exactly how
    many times the embedding model was loaded (must be 1).
    """
    dense = FAISSStore.load(index_dir)
    bm25 = BM25Retriever.load(index_dir)
    if embeddings is None:
        embeddings = EmbeddingService(model_name=model, batch_size=batch_size, show_progress=False)
    hybrid = HybridRetriever(dense=dense, bm25=bm25, embeddings=embeddings)

    rows: list[dict] = []
    for entry in SMOKE_QUERIES:
        query = entry["query"]
        language = entry["language"]
        evidence_ids: set[str] = set()
        if entry["evidence"] is not None:
            evidence_ids = set(_evidence_chunk_ids(dense.chunks, *entry["evidence"]))
        evidence_present = bool(evidence_ids)

        embed_start = time.perf_counter()
        qvec = embeddings.embed_query(query)
        embedding_ms = (time.perf_counter() - embed_start) * 1000.0

        t0 = time.perf_counter()
        dense_results = dense.search(qvec, top_k=top_k, language=language)
        dense_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        bm25_results = bm25.search(query, top_k=top_k, language=language)
        bm25_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        hybrid_results = hybrid.search_embedded(qvec, query, top_k=top_k, language=language)
        hybrid_ms = (time.perf_counter() - t0) * 1000.0

        def _top1(results: list) -> tuple:
            return (results[0].chunk_id, float(results[0].score)) if results else (None, None)

        bm25_top1, bm25_top1_score = _top1(bm25_results)
        dense_top1, dense_top1_score = _top1(dense_results)
        hybrid_top1, hybrid_top1_score = _top1(hybrid_results)

        bm25_rank = dense_rank = hybrid_rank = None
        if evidence_present:
            bm25_rank = _rank_of([r.chunk_id for r in bm25_results], evidence_ids)
            dense_rank = _rank_of([r.chunk_id for r in dense_results], evidence_ids)
            hybrid_rank = _rank_of([r.chunk_id for r in hybrid_results], evidence_ids)
            status = "PASS" if hybrid_rank is not None else "FAIL"
        else:
            status = "NOT_APPLICABLE"

        rows.append(
            {
                "label": entry["label"],
                "query": query,
                "language": language,
                "detected_language": detect_script_language(query),
                "evidence_record_passage": entry["evidence"],
                "evidence_chunk_ids": sorted(evidence_ids) if evidence_ids else None,
                "evidence_present": evidence_present,
                "bm25_top1": bm25_top1,
                "bm25_top1_score": bm25_top1_score,
                "dense_top1": dense_top1,
                "dense_top1_score": dense_top1_score,
                "hybrid_top1": hybrid_top1,
                "hybrid_top1_score": hybrid_top1_score,
                "bm25_rank": bm25_rank,
                "dense_rank": dense_rank,
                "hybrid_rank": hybrid_rank,
                "status": status,
                "embedding_ms": round(embedding_ms, 1),
                "dense_ms": round(dense_ms, 1),
                "bm25_ms": round(bm25_ms, 1),
                "hybrid_ms": round(hybrid_ms, 1),
                "total_ms": round(embedding_ms + dense_ms + bm25_ms + hybrid_ms, 1),
                "hybrid_top1_text": hybrid_results[0].text[:100] if hybrid_results else "",
            }
        )
    return {
        "index_dir": str(index_dir),
        "chunks": dense.index.ntotal,
        "model": model,
        "model_loads": embeddings.load_count,
        "rows": rows,
    }


def print_smoke_table(report: dict) -> None:
    print("")
    print(f"Smoke battery — {len(report['rows'])} queries, index {report['index_dir']}:")
    print("")
    for i, row in enumerate(report["rows"], start=1):
        print(f"[{i}] {row['label']}  {row['query']}")
        print(f"    Language: {row['language']} (detected: {row['detected_language']})")
        if row["evidence_present"]:
            print(
                f"    EVIDENCE_EXISTS=YES ({row['evidence_record_passage']!r}; "
                f"{len(row['evidence_chunk_ids'])} indexed chunk(s))"
            )
        else:
            print("    EVIDENCE_EXISTS=NO (no indexed chunk answers this query — coverage gap)")
        for name, cid, score, rank in (
            ("BM25", row["bm25_top1"], row["bm25_top1_score"], row["bm25_rank"]),
            ("Dense", row["dense_top1"], row["dense_top1_score"], row["dense_rank"]),
            ("Hybrid", row["hybrid_top1"], row["hybrid_top1_score"], row["hybrid_rank"]),
        ):
            if cid is None:
                print(f"    {name:<6} top1: (none)")
            else:
                rank_note = f" — evidence rank: {rank}" if rank is not None else ""
                print(f"    {name:<6} top1: {cid} (score {score:.4f}){rank_note}")
        print(
            f"    Latency: embedding {row['embedding_ms']:.1f} ms | dense {row['dense_ms']:.1f} ms | "
            f"bm25 {row['bm25_ms']:.1f} ms | hybrid {row['hybrid_ms']:.1f} ms | "
            f"total {row['total_ms']:.1f} ms"
        )
        print(f"    RETRIEVAL={row['status']}")
        print("")
    print(f"Model loads: {report['model_loads']} (expected exactly 1 for {len(report['rows'])} queries)")


def main_audit(args: argparse.Namespace) -> int:
    index_dir = Path(args.index_dir or (BASE_DIR / "data" / "indexes" / "recursive"))
    if not (index_dir / "index.faiss").exists():
        logger.error(
            "Index not found at %s — build it with "
            "`python scripts/build_retrieval_index.py --strategy recursive --full`",
            index_dir,
        )
        return 1
    logger.info("Audit index=%s top_k=%d", index_dir, args.audit_top_k)
    report = run_audit(
        index_dir=index_dir,
        model=args.model,
        batch_size=args.batch_size,
        top_k=args.audit_top_k,
    )
    print_audit_table(report)
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)
    args.audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote audit -> %s", args.audit_path)

    # Exit code: 0 when every known-evidence query passed and every
    # missing-evidence query was (correctly) reported as NO-EVIDENCE.
    failed = [r for r in report["rows"] if r["status"] == "FAIL"]
    no_evidence = [r for r in report["rows"] if r["status"] == "NO-EVIDENCE"]
    print("")
    print(f"Known-evidence queries passed: {sum(1 for r in report['rows'] if r['status'] == 'PASS')}")
    print(f"Missing-evidence queries (coverage failure, correctly not retrievable): {len(no_evidence)}")
    if failed:
        print(f"FAILED (evidence present but not retrieved): {[r['label'] for r in failed]}")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation across chunking strategies")
    parser.add_argument("--mode", choices=("benchmark", "full"), default="benchmark",
                        help="benchmark: reproducible 495-record sample manifest scored across "
                             "data/indexes/benchmark/{strategy}; full: all 5,000 records of the "
                             "production index data/indexes/recursive")
    parser.add_argument("--smoke", action="store_true",
                        help="Run the fixed 5-query smoke battery against the mode's index and exit")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--eval-queries", type=int, default=DEFAULT_EVAL_QUERIES)
    parser.add_argument("--rebuild-eval", action="store_true")
    parser.add_argument("--strategy", action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--audit", action="store_true",
                        help="Run the production-index per-query audit table instead of Recall@K")
    parser.add_argument("--index-dir", type=Path, default=None,
                        help="Index dir for --audit / --mode full (default: data/indexes/recursive)")
    parser.add_argument("--audit-top-k", type=int, default=10)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT_PATH)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout
    )

    if args.audit:
        raise SystemExit(main_audit(args))

    config = resolve_mode(args)
    mode = config["mode"]
    sampled_records = config["record_ids"]

    if mode == "full" and args.strategy:
        logger.info("--mode full evaluates the production recursive index; ignoring --strategy=%s",
                    args.strategy)

    # --- pre-model-load index validation ---------------------------------
    # A missing, truncated or mode-mismatched index is a hard error; there is
    # no silent fallback to a different index.
    validated: dict[str, dict] = {}
    if args.smoke:
        validated[config["strategies"][0]] = validate_index(
            config["primary_index"], mode, config["records"]
        )
    else:
        for strategy in config["strategies"]:
            index_dir = config["indexes_dir"] if mode == "full" else config["indexes_dir"] / strategy
            validated[strategy] = validate_index(index_dir, mode, config["records"])
    # The header mirrors the primary index (recursive in both modes).
    primary_info = validated["recursive"] if "recursive" in validated else validated[config["strategies"][0]]

    # Mode facts are printed BEFORE the embedding model is loaded.
    print_mode_header(
        mode=mode,
        manifest_path=config["manifest_path"],
        index_dir=config["primary_index"],
        records=config["records"],
        chunks=primary_info["chunks"],
        languages=primary_info["languages"],
    )

    if args.smoke:
        report = run_smoke(
            index_dir=config["primary_index"],
            model=args.model,
            batch_size=args.batch_size,
            top_k=args.audit_top_k,
        )
        print_smoke_table(report)
        loads = report["model_loads"]
        logger.info(
            "Model loads: %d — single model reused across %d smoke queries (expected 1)",
            loads,
            len(report["rows"]),
        )
        if loads != 1:
            logger.error("Expected exactly 1 model load, got %d", loads)
            sys.exit(1)
        return

    # --- evaluation set --------------------------------------------------
    eval_path = FULL_EVAL_PATH if mode == "full" else Path(args.eval_path)
    corpus_for_queries = config["manifest_path"] if mode == "full" else Path(args.corpus)
    if not eval_path.exists() or args.rebuild_eval:
        queries = build_eval_queries(corpus_for_queries, sampled_records, max_queries=args.eval_queries)
        save_eval_set(queries, eval_path)
    else:
        queries = load_eval_set(eval_path)
        queries = [q for q in queries if q.get("record_id") in sampled_records]
        if len(queries) > args.eval_queries:
            queries = queries[: args.eval_queries]
        if not queries:
            logger.warning("Eval file has no queries from the sampled records; rebuilding")
            queries = build_eval_queries(corpus_for_queries, sampled_records, max_queries=args.eval_queries)
            save_eval_set(queries, eval_path)
    logger.info("Eval set: %d queries (query example: %r)",
                len(queries), queries[0]["query"][:40] if queries else "")

    # One embedding model, created once and reused for every strategy and
    # query — never recreated per query.
    embeddings = EmbeddingService(model_name=args.model, batch_size=args.batch_size)
    logger.info("Embedding model=%s dim=%d", args.model, embeddings.dimension)

    results: dict = {
        "_meta": {
            "mode": mode,
            "dataset": "ai4bharat/MSMARCO-XI",
            "embedding_model": args.model,
            "manifest": config["manifest"],
            "records": config["records"],
            "chunks": primary_info["chunks"],
            "languages": primary_info["languages"],
            "eval_queries": len(queries),
            "recall_definition": (
                "binary Recall@K: fraction of eval queries with >=1 relevant chunk "
                "(record_id, passage_index of a selected passage) in the top-K; "
                "relevant is language-agnostic"
            ),
            "latency": "nearest-rank P50/P70/P100 over per-query timings (ms)",
        },
    }

    header = (
        f"{'Strategy':<12}{'Chunks':>9}{'Dense R@5':>11}{'BM25 R@5':>11}"
        f"{'Hybrid R@5':>12}"
        f"{'Hyb R@1':>10}{'Hyb R@3':>10}{'Hyb R@10':>11}"
    )
    print("")
    print(header)
    print("-" * len(header))

    for strategy in config["strategies"]:
        index_dir = config["indexes_dir"] if mode == "full" else config["indexes_dir"] / strategy
        if not (index_dir / "index.faiss").exists():
            logger.error(
                "Index missing for %s in %s (run scripts/build_retrieval_index.py)",
                strategy,
                index_dir,
            )
            sys.exit(1)

        logger.info("Evaluating strategy %s ...", strategy)
        dense = FAISSStore.load(index_dir)
        bm25 = BM25Retriever.load(index_dir)
        hybrid = HybridRetriever(dense=dense, bm25=bm25, embeddings=embeddings)

        if mode == "full":
            # Full mode: the indexed metadata IS the corpus's chunks.
            relevant = relevant_chunk_ids(queries, dense.chunks)
        else:
            chunk_stream = chunks_for_records(args.chunks_dir / f"{strategy}.jsonl", sampled_records)
            relevant = relevant_chunk_ids(queries, chunk_stream)

        stats = evaluate_strategy(strategy, queries, dense, bm25, hybrid, embeddings, relevant)
        stats["latency_ms"]["modes"] = latency_table(stats)

        results[strategy] = {
            "chunks": dense.index.ntotal,
            **{k: v for k, v in stats.items()},
        }
        d = stats["dense"]
        b = stats["bm25"]
        h = stats["hybrid"]
        print(
            f"{strategy:<12}{dense.index.ntotal:>9}"
            f"{d['recall@5']:>11.3f}{b['recall@5']:>11.3f}{h['recall@5']:>12.3f}"
            f"{h['recall@1']:>10.3f}{h['recall@3']:>10.3f}{h['recall@10']:>11.3f}"
        )

    results["_meta"]["model_loads"] = embeddings.load_count

    results_path = FULL_RESULTS if mode == "full" else Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Wrote %s", results_path)

    # --- selection hint ---------------------------------------------------
    print("")
    print("Per-mode P50 retrieval latency (ms) [embed+search]:")
    print(f"{'Strategy':<12}{'Dense P50':>12}{'BM25 P50':>12}{'Hybrid P50':>12}")
    for strategy in config["strategies"]:
        modes = results[strategy]["latency_ms"]["modes"]
        print(
            f"{strategy:<12}{modes['dense']['p50']:>12.1f}{modes['bm25']['p50']:>12.1f}"
            f"{modes['hybrid']['p50']:>12.1f}"
        )

    print("")
    logger.info("Done. Results -> %s", results_path)


if __name__ == "__main__":
    main()

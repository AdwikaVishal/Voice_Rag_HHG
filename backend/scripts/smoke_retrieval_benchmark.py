"""Segment 4 — retrieval smoke benchmark on the production retriever.

Loads the production hybrid retriever (FAISS + BM25 + RRF, ``recursive``
strategy by default) and times a small batch of real queries (first 75 entries
of ``data/processed/retrieval_eval.jsonl`` by default) end to end — including
query embedding, dense search, BM25 search and RRF. Reports mean / median /
P95 / P99 latency and queries-per-second, and writes a machine-readable copy
to ``data/processed/retrieval_smoke_benchmark.json``.

No index building or re-embedding is performed.

Usage (from backend/):
    python scripts/smoke_retrieval_benchmark.py [--n 75] [--strategy recursive]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import CHUNKING_STRATEGY  # noqa: E402
from app.retrieval import ProductionRetriever, load_chunk_records  # noqa: E402

logging.basicConfig(level=logging.WARNING)

DEFAULT_EVAL_QUERIES = BASE_DIR / "data" / "processed" / "retrieval_eval.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "data" / "processed" / "retrieval_smoke_benchmark.json"
DEFAULT_N = 75


def nearest_rank(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int((percentile / 100.0) * len(sorted_values))
    return sorted_values[min(idx, len(sorted_values) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default=CHUNKING_STRATEGY)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="number of queries")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-queries", type=Path, default=DEFAULT_EVAL_QUERIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    queries = [row["query"] for row in load_chunk_records(args.eval_queries)][: args.n]
    if len(queries) < args.n:
        print(f"NOTE: only {len(queries)} queries available (requested {args.n})")
    if not queries:
        print(f"ERROR: no queries in {args.eval_queries}")
        raise SystemExit(1)

    retriever = ProductionRetriever(strategy=args.strategy)
    n = len(queries)

    # Warm-up: the first query pays the model + index load cost. It is timed
    # separately so the latency stats reflect steady-state behavior.
    warm_start = time.perf_counter()
    retriever.search(queries[0], top_k=args.top_k)
    warmup_ms = (time.perf_counter() - warm_start) * 1000.0

    timings: list[float] = []
    for i, query in enumerate(queries):
        start = time.perf_counter()
        results = retriever.search(query, top_k=args.top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings.append(elapsed_ms)
        if not results:
            print(f"WARN: query #{i} returned no results: {query[:40]!r}")

    sorted_timings = sorted(timings)
    report = {
        "strategy": retriever.strategy,
        "index_chunks": retriever.chunk_count,
        "queries": n,
        "top_k": args.top_k,
        "timing_unit": "ms/query (embed + dense + bm25 + rrf)",
        "warmup_ms": round(warmup_ms, 2),
        "total_seconds": round(sum(timings) / 1000.0, 3),
        "queries_per_second": round(n / (sum(timings) / 1000.0), 2),
        "mean_ms": round(statistics.mean(timings), 2),
        "median_ms": round(statistics.median(timings), 2),
        "p50_ms": round(nearest_rank(sorted_timings, 50), 2),
        "p70_ms": round(nearest_rank(sorted_timings, 70), 2),
        "p95_ms": round(nearest_rank(sorted_timings, 95), 2),
        "p99_ms": round(nearest_rank(sorted_timings, 99), 2),
        "max_ms": round(max(timings), 2),
        "per_query_ms": [round(t, 2) for t in timings],
    }

    print(f"Strategy:        {retriever.strategy}")
    print(f"Index chunks:    {retriever.chunk_count}")
    print(f"Queries:         {n}  (top_k={args.top_k})")
    print(f"Warm-up (load):  {report['warmup_ms']} ms (model + index load, not in stats)")
    print(f"Total:           {report['total_seconds']}s")
    print(f"Throughput:      {report['queries_per_second']} queries/sec")
    print(f"Latency:         mean={report['mean_ms']} median={report['median_ms']} "
          f"P95={report['p95_ms']} P99={report['p99_ms']} max={report['max_ms']} ms")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote -> {args.output}")


if __name__ == "__main__":
    main()

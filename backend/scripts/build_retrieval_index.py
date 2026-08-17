"""Segment 3 — build dense (FAISS) + sparse (BM25) retrieval indexes.

The default command builds a fast, deterministic *benchmark* index — a
~10,000-chunk sample per strategy drawn from the same source records — so the
retrieval comparison is fair without embedding ~400k chunks on CPU. Use the
``--full`` flag only to scale the winning strategy to the entire corpus.

Pipeline (benchmark mode):
    1. run the embedding benchmark (100 / 500 / 1,000 texts) and write
       ``data/processed/embedding_benchmark.json``;
    2. unless the measured throughput says otherwise, target ``--sample-size``
       chunks per strategy and automatically fall back to a smaller sample if
       10k would take too long on this machine;
    3. draw a deterministic set of source records (``--seed``) and keep every
       chunk of those records for each strategy (fair + reproducible);
    4. embed the sample with a visible progress bar, build exact ``IndexFlatIP``
       (normalized embeddings -> cosine) + ``BM25Okapi``, persist to
       ``data/indexes/benchmark/{strategy}/``;
    5. write ``data/processed/benchmark_manifest.json`` so the eval script can
       reuse the exact same record set and queries.

Usage (from backend/):
    python scripts/build_retrieval_index.py
    python scripts/build_retrieval_index.py --sample-size 10000 --batch-size 32
    python scripts/build_retrieval_index.py --strategy recursive --full
    python scripts/build_retrieval_index.py --strategy fixed --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.chunking.factory import available_strategies  # noqa: E402
from app.retrieval import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    EmbeddingService,
    FAISSStore,
    build_manifest,
    chunks_for_records,
    count_chunks,
    load_chunk_records,
    record_list_from_corpus,
)
from app.retrieval.bm25 import BM25Retriever  # noqa: E402

logger = logging.getLogger("build_retrieval_index")

DEFAULT_CHUNKS_DIR = BASE_DIR / "data" / "processed" / "chunks"
DEFAULT_CORPUS = BASE_DIR / "data" / "processed" / "msmarco_xi_sample.jsonl"
DEFAULT_INDEXES_DIR = BASE_DIR / "data" / "indexes"
DEFAULT_MANIFEST = BASE_DIR / "data" / "processed" / "benchmark_manifest.json"
DEFAULT_BENCHMARK_PATH = BASE_DIR / "data" / "processed" / "embedding_benchmark.json"

EMBED_BENCH_SIZES = (100, 500, 1000)
DEFAULT_MAX_EMBED_SECONDS = 1800.0


# --------------------------------------------------------------------------
# Embedding benchmark
# --------------------------------------------------------------------------

def run_embedding_benchmark(
    service: EmbeddingService,
    chunk_path: Path,
    out_path: Path,
    sample_size: int,
) -> dict:
    """Measure texts/sec on 100 / 500 / 1,000 chunks and persist the result."""
    texts = [c["text"] for c in load_chunk_records(chunk_path)][: max(EMBED_BENCH_SIZES)]
    logger.info(
        "Embedding benchmark: %d texts (model=%s, batch=%d)",
        len(texts),
        service.model_name,
        service.batch_size,
    )
    entries = []
    for n in EMBED_BENCH_SIZES:
        if n > len(texts):
            logger.warning("Only %d texts available; skipping size %d", len(texts), n)
            continue
        t0 = time.perf_counter()
        service.embed_documents(texts[:n], show_progress=False)
        dt = time.perf_counter() - t0
        entries.append(
            {
                "texts": n,
                "total_seconds": round(dt, 3),
                "texts_per_second": round(n / dt, 2) if dt else 0.0,
                "milliseconds_per_text": round(dt * 1000.0 / n, 2) if dt else 0.0,
            }
        )
        logger.info(
            "  embedded %d texts in %.1fs -> %.1f texts/sec",
            n,
            dt,
            n / dt if dt else 0.0,
        )
    decision = {"reduced_sample": False, "reason": ""}
    largest = entries[-1]
    tps = largest["texts_per_second"]
    if tps > 0:
        decision["estimated_sample_seconds"] = round(sample_size / tps, 1)
    doc = {
        "model": service.model_name,
        "batch_size": service.batch_size,
        "device": service.device,
        "normalized": True,
        "target_sample_size": sample_size,
        "results": entries,
        "note": (
            "steady-state throughput measured on the fixed-strategy chunk texts; "
            "the first (smallest) entry includes model load + warmup"
        ),
        "auto_reduce": decision,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote embedding benchmark -> %s", out_path)
    return doc


# --------------------------------------------------------------------------
# Index building
# --------------------------------------------------------------------------

def strategy_out_dir(indexes_dir: Path, strategy: str, full: bool) -> Path:
    if full:
        return indexes_dir / strategy
    return indexes_dir / "benchmark" / strategy


def already_built(out_dir: Path) -> bool:
    return (
        (out_dir / "index.faiss").exists()
        and (out_dir / "metadata.json").exists()
        and (out_dir / "bm25.pkl").exists()
    )


def build_strategy(
    strategy: str,
    chunk_path: Path,
    out_dir: Path,
    service: EmbeddingService,
    batch_size: int,
    record_ids: list[str] | None,
    limit: int | None,
    force: bool,
) -> int:
    """Embed + index one strategy's chunks and persist FAISS + BM25."""
    if out_dir.exists() and not force and already_built(out_dir):
        logger.info("Skipping %s (already built in %s)", strategy, out_dir)
        return 0

    t0 = time.perf_counter()
    if record_ids is not None:
        chunks = list(chunks_for_records(chunk_path, record_ids))
        logger.info("[%s] selecting chunks for %d sampled records", strategy, len(record_ids))
    else:
        chunks = list(load_chunk_records(chunk_path))
    if limit:
        chunks = chunks[:limit]
    if not chunks:
        logger.error("[%s] no chunks to index; aborting strategy", strategy)
        return 0
    logger.info("[%s] embedding %d chunks (batch=%d)...", strategy, len(chunks), batch_size)
    matrix = service.embed_documents([c["text"] for c in chunks], show_progress=True)
    if matrix.shape[0] != len(chunks):
        raise ValueError(f"[{strategy}] embedded {matrix.shape[0]} but loaded {len(chunks)} chunks")

    dense = FAISSStore.build(chunks, matrix, model_name=service.model_name)
    bm25 = BM25Retriever.build(chunks)
    out_dir.mkdir(parents=True, exist_ok=True)
    dense.save(out_dir)
    bm25.save(out_dir)
    logger.info(
        "[%s] built %d chunks in %.1fs -> %s",
        strategy,
        len(chunks),
        time.perf_counter() - t0,
        out_dir,
    )
    return len(chunks)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 retrieval indexes")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--embedding-benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-size", type=int, default=10000, help="Target chunks per strategy (benchmark mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-embed-seconds", type=float, default=DEFAULT_MAX_EMBED_SECONDS,
                        help="Auto-reduce the sample if embedding it is estimated to take longer")
    parser.add_argument("--limit", type=int, default=None, help="Cap chunks per strategy (smoke testing)")
    parser.add_argument("--strategy", action="append", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--full", action="store_true",
                        help="Build the FULL corpus index for --strategy (not the benchmark sample)")
    parser.add_argument("--skip-embedding-benchmark", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    strategies = sorted(set(args.strategy or available_strategies()))
    logger.info("Strategies: %s", ", ".join(strategies))

    service = EmbeddingService(model_name=args.model, batch_size=args.batch_size)
    logger.info(
        "Embedding model=%s dim=%d batch=%d",
        args.model,
        service.dimension,
        service.batch_size,
    )

    # --- embedding benchmark --------------------------------------------
    benchmark_doc = None
    if not args.skip_embedding_benchmark:
        bench_chunk = args.chunks_dir / "fixed.jsonl"
        benchmark_doc = run_embedding_benchmark(
            service, bench_chunk, args.embedding_benchmark, args.sample_size
        )
    else:
        logger.info("Skipping embedding benchmark (--skip-embedding-benchmark)")

    # --- full-corpus mode ------------------------------------------------
    if args.full:
        if not args.strategy:
            parser.error("--full requires --strategy (scale only the winning strategy)")
        for strategy in strategies:
            out_dir = strategy_out_dir(args.indexes_dir, strategy, full=True)
            build_strategy(
                strategy,
                args.chunks_dir / f"{strategy}.jsonl",
                out_dir,
                service,
                args.batch_size,
                record_ids=None,
                limit=args.limit,
                force=args.force,
            )
        logger.info("Done (full mode).")
        return

    # --- benchmark mode --------------------------------------------------
    record_ids = record_list_from_corpus(args.corpus)
    if not record_ids:
        logger.error("No records found in corpus %s", args.corpus)
        sys.exit(1)
    total_chunks_fixed = count_chunks(args.chunks_dir / "fixed.jsonl")
    logger.info(
        "Corpus: %d records, fixed strategy produces %d chunks (~%.1f chunks/record)",
        len(record_ids),
        total_chunks_fixed,
        total_chunks_fixed / len(record_ids) if record_ids else 0.0,
    )

    sample_size = args.sample_size
    if benchmark_doc is not None:
        tps = benchmark_doc["results"][-1]["texts_per_second"]
        if tps > 0:
            estimate = sample_size / tps
            logger.info(
                "Estimated %.1fs to embed %d chunks @ %.1f texts/sec (max %.0fs)",
                estimate, sample_size, tps, args.max_embed_seconds,
            )
            while estimate > args.max_embed_seconds and sample_size > 1000:
                sample_size //= 2
                estimate = sample_size / tps
                logger.warning("  reducing sample to %d chunks (estimated %.1fs)", sample_size, estimate)
            if estimate > args.max_embed_seconds:
                logger.warning(
                    "Even %d chunks estimated at %.1fs — building anyway; expect a long run "
                    "(interrupt safely: indexes are saved per strategy)",
                    sample_size, estimate,
                )

    sampled, manifest = build_manifest(
        record_ids=record_ids,
        target_sample_size=args.sample_size,
        total_chunks_fixed=total_chunks_fixed,
        sample_size=sample_size,
        seed=args.seed,
        strategies=strategies,
    )
    manifest["final_sample_size"] = sample_size
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Sampled %d/%d records (seed=%d) -> ~%d chunks/strategy; manifest -> %s",
        len(sampled),
        len(record_ids),
        args.seed,
        sample_size,
        args.manifest,
    )

    for strategy in strategies:
        out_dir = strategy_out_dir(args.indexes_dir, strategy, full=False)
        build_strategy(
            strategy,
            args.chunks_dir / f"{strategy}.jsonl",
            out_dir,
            service,
            args.batch_size,
            record_ids=sampled,
            limit=args.limit,
            force=args.force,
        )

    logger.info("Done. Benchmark indexes under %s", args.indexes_dir / "benchmark")


if __name__ == "__main__":
    main()

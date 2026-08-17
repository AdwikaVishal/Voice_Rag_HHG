"""Segment 2 — chunking benchmark over the processed MSMARCO-XI sample.

Runs every chunking strategy over the corpus, records structural chunk
statistics (chunk counts, token-length distribution, chunks per passage,
processing time), writes one JSONL of chunks per strategy and a single
results JSON.

No embeddings, retrieval, or network calls are made; the run is local and
deterministic.

Usage (from backend/):
    python scripts/benchmark_chunking.py
    python scripts/benchmark_chunking.py --limit 500 --chunk-size 200 --overlap 0.2
    python scripts/benchmark_chunking.py --text-field translated
    python scripts/benchmark_chunking.py --strategy fixed --strategy sentence
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.chunking.factory import available_strategies, create_chunker  # noqa: E402

logger = logging.getLogger("benchmark_chunking")

DEFAULT_INPUT = BASE_DIR / "data" / "processed" / "msmarco_xi_sample.jsonl"
DEFAULT_CHUNKS_DIR = BASE_DIR / "data" / "processed" / "chunks"
DEFAULT_RESULTS = BASE_DIR / "data" / "processed" / "chunking_results.json"


def iter_records(path: Path):
    """Yield one parsed record per JSONL line, streaming."""
    with path.open(encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunking benchmark over processed MSMARCO-XI")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument(
        "--text-field",
        choices=("english", "translated", "both"),
        default="both",
        help="Which passage field(s) to chunk (default: both)",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        default=None,
        help="Run only this strategy (repeatable); default: all",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N records")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if not args.input.exists():
        logger.error("Input not found: %s", args.input)
        sys.exit(1)

    strategies = sorted(set(args.strategy or available_strategies()))
    chunkers = {
        name: create_chunker(
            name,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            text_field=args.text_field,
        )
        for name in strategies
    }
    logger.info(
        "Strategies: %s | chunk_size=%d overlap=%.2f text_field=%s",
        ", ".join(strategies),
        args.chunk_size,
        args.overlap,
        args.text_field,
    )

    args.chunks_dir.mkdir(parents=True, exist_ok=True)

    per_strategy = {
        name: {
            "chunks": 0,
            "passages": 0,
            "token_counts": [],
            "chunks_per_passage": [],
        }
        for name in strategies
    }
    timings = {name: 0.0 for name in strategies}

    records = 0
    passages_total = 0
    languages: set[str] = set()

    out_files: dict[str, object] = {}
    try:
        for name in strategies:
            out_files[name] = (args.chunks_dir / f"{name}.jsonl").open("w", encoding="utf-8")

        for record in iter_records(args.input):
            records += 1
            passages = record.get("passages") or []
            passages_total += len(passages)
            if record.get("source_lang"):
                languages.add(record["source_lang"])
            if record.get("target_lang"):
                languages.add(record["target_lang"])

            for name in strategies:
                chunker = chunkers[name]
                t0 = time.perf_counter()
                chunks = chunker.chunk(record)
                timings[name] += time.perf_counter() - t0

                per_strategy[name]["passages"] += len(passages)
                per_strategy[name]["chunks"] += len(chunks)
                per_strategy[name]["chunks_per_passage"].extend(
                    Counter(c.passage_index for c in chunks).values()
                )
                for chunk in chunks:
                    per_strategy[name]["token_counts"].append(chunk.token_count)
                    out_files[name].write(
                        json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n"
                    )

            if args.limit and records >= args.limit:
                break
    finally:
        for handle in out_files.values():
            handle.close()

    logger.info(
        "Processed %d records, %d passages | languages=%s",
        records,
        passages_total,
        sorted(l for l in languages if l),
    )

    results = {
        "dataset": "ai4bharat/MSMARCO-XI",
        "input_records": records,
        "input_passages": passages_total,
        "languages_observed": sorted(l for l in languages if l),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "text_field": args.text_field,
        "strategies": {},
    }

    header = (
        f"{'Strategy':<12}{'Records':>9}{'Passages':>10}{'Chunks':>9}"
        f"{'Avg Tok':>9}{'Med Tok':>9}{'Min':>5}{'Max':>6}"
        f"{'Chk/Pass':>9}{'Time ms':>10}"
    )
    print("")
    print(header)
    print("-" * len(header))

    for name in strategies:
        stat = per_strategy[name]
        counts = stat["token_counts"]
        cpp = stat["chunks_per_passage"]
        avg = statistics.mean(counts) if counts else 0.0
        median = statistics.median(counts) if counts else 0
        minv = min(counts) if counts else 0
        maxv = max(counts) if counts else 0
        cpp_avg = statistics.mean(cpp) if cpp else 0.0
        cpp_median = statistics.median(cpp) if cpp else 0
        elapsed_ms = round(timings[name] * 1000, 3)

        results["strategies"][name] = {
            "records": records,
            "passages": stat["passages"],
            "chunks": stat["chunks"],
            "avg_tokens": round(avg, 2),
            "median_tokens": median,
            "min_tokens": minv,
            "max_tokens": maxv,
            "chunks_per_passage_avg": round(cpp_avg, 3),
            "chunks_per_passage_median": cpp_median,
            "processing_ms": elapsed_ms,
        }
        print(
            f"{name:<12}{records:>9}{stat['passages']:>10}{stat['chunks']:>9}"
            f"{avg:>9.2f}{median:>9}{minv:>5}{maxv:>6}{cpp_avg:>9.3f}{elapsed_ms:>10.1f}"
        )

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Wrote %s", args.results.relative_to(BASE_DIR))
    for name in strategies:
        logger.info("Wrote %s", (args.chunks_dir / f"{name}.jsonl").relative_to(BASE_DIR))


if __name__ == "__main__":
    main()

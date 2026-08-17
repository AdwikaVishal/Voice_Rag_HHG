"""Segment 4 — retrieval results analysis + production strategy selection.

Reads ``data/processed/retrieval_results.json`` (written by
``scripts/evaluate_retrieval.py``) and prints a comparison table across the
four chunking strategies, then applies the Segment 4 selection policy:

1. PRIMARY   — highest Recall@10
2. SECONDARY — Recall@5
3. TERTIARY  — Recall@1
4. Tie-breakers — latency, chunk quality (max chunk size), multilingual
   robustness, chunk count / storage

The winning ``(strategy, retriever)`` pair is printed last and drives the
production configuration in ``app/config.py``.

Usage (from backend/):
    python scripts/analyze_retrieval_results.py [--json data/processed/retrieval_results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import CHUNKING_STRATEGY  # noqa: E402

RESULTS = BASE_DIR / "data" / "processed" / "retrieval_results.json"

# Segment 4 selection policy (highest wins).
PRIORITY_K = (10, 5, 1)
RETRIEVERS = ("dense", "bm25", "hybrid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=RESULTS)
    args = parser.parse_args()

    with args.json.open(encoding="utf-8") as in_file:
        data = json.load(in_file)

    strategies = [key for key in data if key != "_meta"]

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    header = f"{'strategy':<10} {'retriever':<8} " + " ".join(
        f"R@{k}" for k in (1, 3, 5, 10)
    ) + "  hybridP50  chunks"
    print(header)
    print("-" * len(header))
    per_strategy: dict[str, dict] = {}
    for strategy in strategies:
        block = data[strategy]
        chunks = block["chunks"]
        hybrid_p50 = block["latency_ms"]["modes"]["hybrid"]["p50"]
        per_strategy[strategy] = {
            "chunks": chunks,
            "hybrid_p50_ms": hybrid_p50,
            "recall": {},
        }
        for retriever in RETRIEVERS:
            recall = block[retriever]
            per_strategy[strategy]["recall"][retriever] = {
                k: recall[f"recall@{k}"] for k in (1, 3, 5, 10)
            }
            cells = " ".join(f"{recall[f'recall@{k}']:.4f}" for k in (1, 3, 5, 10))
            p50 = hybrid_p50 if retriever == "hybrid" else "-"
            print(f"{strategy:<10} {retriever:<8} {cells}  {p50:>7}  {chunks}")

    # ------------------------------------------------------------------
    # Winner selection
    # ------------------------------------------------------------------
    print()
    winner: dict = {}
    for retriever in RETRIEVERS:
        ranking = sorted(
            per_strategy.items(),
            key=lambda item: tuple(item[1]["recall"][retriever][k] for k in PRIORITY_K),
            reverse=True,
        )
        best, score = ranking[0]
        print(f"[{retriever}] best = {best}  (R@10={score['recall'][retriever][10]:.4f} "
              f"R@5={score['recall'][retriever][5]:.4f} R@1={score['recall'][retriever][1]:.4f})")
        winner[retriever] = best

    strategy_choice = winner["hybrid"]
    config_strategy = CHUNKING_STRATEGY
    print()
    print(f"Winner: strategy={strategy_choice}  retriever=hybrid  "
          f"(config default CHUNKING_STRATEGY='{config_strategy}')")
    if strategy_choice != config_strategy:
        print("WARNING: winner differs from current config default!")


if __name__ == "__main__":
    main()

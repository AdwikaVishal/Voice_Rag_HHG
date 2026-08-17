"""Retrieval evaluation helpers (Recall@K, latency percentiles, ground truth).

Ground truth mapping
--------------------
MSMARCO-XI exposes, per query record, a list of ``passages`` where
``is_selected == 1`` marks the passages that answer the query. Segment 2's
chunk records keep a stable ``passage_index`` from the source passage, so a
chunk is *relevant* to a query iff ``(record_id, passage_index)`` is a
selected passage of that query. Relevance is deliberately language-agnostic:
the English and Urdu chunks of a selected passage both count, because they
represent the same source passage (the PRD cares about retrieving the passage,
not a specific script).

No relevance labels are fabricated — queries without any selected passage are
excluded, and every query used for scoring has a non-empty relevant set.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

DEFAULT_RECALL_KS = (1, 3, 5, 10)
DEFAULT_PERCENTILES = (50, 70, 100)


# --------------------------------------------------------------------------
# Latency percentiles
# --------------------------------------------------------------------------

def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile of ``values`` (``p`` in 0..100).

    Empty input raises :class:`ValueError`; a single value returns itself.
    Deterministic and cheap for the modest query counts we benchmark.
    """
    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(p / 100.0 * len(ordered))))
    return float(ordered[rank - 1])


def percentiles(
    values: Sequence[float],
    ps: Sequence[int] = DEFAULT_PERCENTILES,
) -> dict[str, float]:
    """Map ``"p50": ...`` style keys to nearest-rank percentiles of ``values``."""
    return {f"p{p}": round(percentile(values, p), 3) for p in ps}


def summary_stats(values: Sequence[float]) -> dict[str, float]:
    """P50 / P70 / P100 plus mean / median for a latency bucket."""
    if not values:
        return {}
    out = percentiles(values)
    out["mean_ms"] = round(statistics.mean(values), 3)
    out["median_ms"] = round(statistics.median(values), 3)
    out["count"] = len(values)
    return out


# --------------------------------------------------------------------------
# Evaluation set
# --------------------------------------------------------------------------

def build_eval_queries(
    corpus_path: Path,
    record_ids: Iterable[str],
    max_queries: int = 500,
) -> list[dict]:
    """Build the reproducible evaluation set from records in ``record_ids``.

    Keeps, in corpus order, records that (a) belong to the sampled record set
    and (b) have at least one ``is_selected == 1`` passage. ``max_queries``
    caps the size (default 500, per the PRD range of 200-500).
    """
    wanted = set(record_ids)
    queries: list[dict] = []
    with Path(corpus_path).open(encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("record_id") not in wanted:
                continue
            selected = sorted(
                p["passage_index"]
                for p in (record.get("passages") or [])
                if p.get("is_selected") == 1
            )
            if not selected:
                continue
            queries.append(
                {
                    "query_id": record.get("query_id"),
                    "record_id": record["record_id"],
                    "query": record.get("query", ""),
                    "english_query": record.get("english_query", ""),
                    "query_type": record.get("query_type"),
                    "relevant_passage_indices": selected,
                }
            )
            if len(queries) >= max_queries:
                break
    return queries


def save_eval_set(queries: list[dict], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as out_file:
        for query in queries:
            out_file.write(json.dumps(query, ensure_ascii=False) + "\n")


def load_eval_set(path: Path) -> list[dict]:
    queries = []
    with Path(path).open(encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


# --------------------------------------------------------------------------
# Ground truth: source passage -> chunks
# --------------------------------------------------------------------------

def relevant_chunk_ids(
    queries: Sequence[dict],
    chunks: Iterable[dict[str, Any]],
) -> dict[Any, set[str]]:
    """Map query_id -> set of relevant chunk_ids for the given chunk stream.

    ``queries`` must carry ``record_id`` and ``relevant_passage_indices``.
    ``chunks`` is any iterable of chunk records (typically a filtered stream
    that only contains sampled records). A chunk is relevant to a query when
    its ``(record_id, passage_index)`` pair is one of that query's selected
    passages.
    """
    record_to_query = {q["record_id"]: q for q in queries}
    relevant: dict[Any, set[str]] = defaultdict(set)
    for chunk in chunks:
        query = record_to_query.get(chunk.get("record_id"))
        if query is None:
            continue
        if chunk.get("passage_index") in query["relevant_passage_indices"]:
            relevant[query["query_id"]].add(chunk["chunk_id"])
    return dict(relevant)


def source_passage_chunk_ids(
    chunk_path: Path,
    record_id: str,
    passage_index: int,
) -> list[str]:
    """All chunk_ids in ``chunk_path`` covering one source passage."""
    return [
        chunk["chunk_id"]
        for chunk in _iter_path(chunk_path)
        if chunk.get("record_id") == record_id and chunk.get("passage_index") == passage_index
    ]


def _iter_path(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary Recall@K: 1 if any of the top-K is relevant, else 0."""
    if k < 1:
        return 0.0
    return 1.0 if any(cid in relevant for cid in retrieved[:k]) else 0.0


def aggregate_recall(
    hits: Sequence[Sequence[float]],
    ks: Sequence[int] = DEFAULT_RECALL_KS,
) -> dict[str, float]:
    """Mean binary recall per K over per-query hit rows.

    Each row of ``hits`` is one query's ``[recall@k for k in ks]``.
    """
    if not hits:
        return {f"recall@{k}": 0.0 for k in ks}
    columns = list(zip(*hits))
    return {
        f"recall@{k}": round(statistics.mean(columns[i]), 4)
        for i, k in enumerate(ks)
    }

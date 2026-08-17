"""Deterministic, strategy-fair benchmark sampling.

Segment 3 must evaluate chunking strategies against the *same* underlying
source records, otherwise the retrieval comparison is unfair (a sample of
records for one strategy vs a different sample for another). We therefore
sample source records once and feed the same record set to every strategy.

Method
------
1. Collect the ordered record list from the processed corpus
   (``data/processed/msmarco_xi_sample.jsonl``).
2. Count the total chunks produced by the baseline strategy (``fixed``) so we
   can convert a chunk-count target into a record-count target.
3. Draw ``k`` records without replacement using a seeded RNG
   (``random.Random(seed).sample``). ``k`` is chosen so the ``fixed`` strategy
   lands as close to ``sample_size`` as possible.
4. Every strategy then contributes all chunks that belong to those records.

Language balance is preserved implicitly: every record chunked by Segment 2
produces both an ``eng_Latn`` and a ``urd_Arab`` chunk for each passage, so a
record-uniform sample is automatically ~50/50 across languages. The seed
makes the sample reproducible.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .models import load_chunk_records

DEFAULT_SEED = 42


def record_list_from_corpus(corpus_path: Path) -> list[str]:
    """Ordered distinct record_ids from the processed corpus (in file order)."""
    record_ids: list[str] = []
    seen: set[str] = set()
    with Path(corpus_path).open(encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            if not line:
                continue
            record_id = json.loads(line).get("record_id")
            if record_id is not None and record_id not in seen:
                seen.add(record_id)
                record_ids.append(str(record_id))
    return record_ids


def count_chunks(chunk_path: Path) -> int:
    """Count JSONL lines in a chunk file without loading them."""
    n = 0
    with Path(chunk_path).open(encoding="utf-8") as in_file:
        for _line in in_file:
            n += 1
    return n


def choose_record_sample_size(
    target_chunks: int,
    n_records: int,
    total_chunks: int,
    min_size: int = 1,
) -> int:
    """Convert a chunk-count target into the record count for a fair sample.

    ``k = round(target_chunks * n_records / total_chunks)``, clamped so the
    sample is never empty and never exceeds the available records. This keeps
    the sampled record set identical across strategies while every strategy
    lands near ``target_chunks`` chunks.
    """
    if n_records <= 0 or total_chunks <= 0:
        return 0
    k = round(target_chunks * n_records / total_chunks)
    return max(min_size, min(k, n_records))


def sample_record_ids(
    record_ids: Sequence[str],
    k: int,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    """Deterministically draw ``k`` records without replacement."""
    if k <= 0 or not record_ids:
        return []
    rng = random.Random(seed)
    return list(rng.sample(list(record_ids), min(k, len(record_ids))))


def chunks_for_records(chunk_path: Path, record_ids: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield chunk records whose ``record_id`` is in the sampled set."""
    wanted = set(record_ids)
    for chunk in load_chunk_records(chunk_path):
        if chunk.get("record_id") in wanted:
            yield chunk


def build_manifest(
    *,
    record_ids: Sequence[str],
    target_sample_size: int,
    total_chunks_fixed: int,
    sample_size: int,
    seed: int = DEFAULT_SEED,
    strategies: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    """Build the sampled record set plus a JSON-serializable manifest.

    The manifest documents exactly how the sample was drawn so the benchmark
    stays reproducible and the eval stage knows which records to use.
    """
    k = choose_record_sample_size(sample_size, len(record_ids), total_chunks_fixed)
    sampled = sample_record_ids(record_ids, k, seed=seed)
    manifest = {
        "seed": seed,
        "target_sample_size": target_sample_size,
        "sample_size": sample_size,
        "records_sampled": len(sampled),
        "total_records": len(record_ids),
        "total_chunks_fixed": total_chunks_fixed,
        "strategies": list(strategies),
        "sampling": (
            "seeded random.sample (seed=%d) of record_ids; all chunks of each "
            "sampled record are included for every strategy, so strategies are "
            "evaluated on the same underlying source records; language balance "
            "is preserved because each record yields eng_Latn + urd_Arab chunks"
            % seed
        ),
        "record_ids": sampled,
    }
    return sampled, manifest

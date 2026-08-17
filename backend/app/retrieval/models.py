"""Typed models and shared data helpers for the retrieval module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

from pydantic import BaseModel


class RetrievalResult(BaseModel):
    """One retrieved chunk, shared by dense / sparse / hybrid retrievers.

    Only stable chunk identifiers are exposed to the rest of the app — raw
    FAISS vector positions are never surfaced.
    """

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = {}
    rank: int = 0
    selected_by_hybrid: bool = False


def load_chunk_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one chunk record (as a plain dict) per JSONL line."""
    with Path(path).open(encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def chunk_to_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Build the retrieval-facing metadata dict for a chunk record.

    The full chunk record is preserved (chunk_id, record_id, query_id,
    languages, passage info, prev/next links, ...) so downstream stages can
    recover everything required without a second lookup.
    """
    return dict(chunk)


def make_result(chunk: dict[str, Any], score: float, rank: int) -> RetrievalResult:
    """Build a :class:`RetrievalResult` from a chunk record + score."""
    return RetrievalResult(
        chunk_id=chunk["chunk_id"],
        score=float(score),
        text=chunk.get("text", ""),
        metadata=chunk_to_metadata(chunk),
        rank=rank,
    )


def chunk_language(chunk: dict[str, Any]) -> Optional[str]:
    """Language of a chunk: ``language`` field, falling back to source/target."""
    lang = chunk.get("language")
    if lang:
        return str(lang)
    return chunk.get("source_lang") or chunk.get("target_lang")

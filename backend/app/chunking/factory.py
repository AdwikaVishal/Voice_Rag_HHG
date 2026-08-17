"""Chunker factory — lets the pipeline switch strategies without rewiring."""

from __future__ import annotations

from typing import Any

from .base import BaseChunker

_SUPPORTED = ("fixed", "sentence", "recursive", "metadata")


def create_chunker(strategy: str, **kwargs: Any) -> BaseChunker:
    """Instantiate a chunker by name.

    Raises ``ValueError`` with a clear message for unknown strategies so
    callers always see the supported set.
    """
    key = (strategy or "").strip().lower()
    if key == "fixed":
        from .fixed import FixedSizeChunker

        return FixedSizeChunker(**kwargs)
    if key == "sentence":
        from .sentence import SentenceChunker

        return SentenceChunker(**kwargs)
    if key == "recursive":
        from .recursive import RecursiveChunker

        return RecursiveChunker(**kwargs)
    if key == "metadata":
        from .metadata import MetadataAwareChunker

        return MetadataAwareChunker(**kwargs)
    raise ValueError(
        f"Unknown chunking strategy {strategy!r}. "
        f"Supported strategies: {', '.join(_SUPPORTED)}"
    )


def available_strategies() -> list[str]:
    """Return the supported strategy names."""
    return list(_SUPPORTED)

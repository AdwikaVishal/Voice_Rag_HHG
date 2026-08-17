"""Strategy 4 — metadata-aware chunking for the multilingual corpus.

Delegates text splitting to a sensible base strategy (sentence-aware by
default) and relies on the shared pipeline to attach retrieval-relevant
metadata to every chunk:

* ``record_id``, ``query_id``
* ``source_lang``, ``target_lang``, ``language``
* ``chunk_position``, ``total_chunks``
* ``prev_chunk_id``, ``next_chunk_id``

This is the strategy designed for the multilingual retrieval stage, where
chunks can be filtered or boosted by language. No vector retrieval is
performed here.
"""

from __future__ import annotations

from .base import BaseChunker

_VALID_BASE = ("fixed", "sentence", "recursive")


class MetadataAwareChunker(BaseChunker):
    """Sentence-aware (default) splitting enriched with language metadata."""

    strategy_name = "metadata"

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: float = 0.0,
        text_field: str = "both",
        base: str = "sentence",
    ):
        super().__init__(chunk_size=chunk_size, overlap=overlap, text_field=text_field)
        if base not in _VALID_BASE:
            raise ValueError(
                f"MetadataAwareChunker base must be one of {_VALID_BASE}, got {base!r}"
            )
        self.base_strategy = base
        self._base_chunker = None

    def _ensure_base(self) -> BaseChunker:
        if self._base_chunker is None:
            from .factory import create_chunker  # lazy import avoids a cycle

            self._base_chunker = create_chunker(
                self.base_strategy,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                text_field=self.text_field,
            )
        return self._base_chunker

    def split_text(self, text: str) -> list[str]:
        return self._ensure_base().split_text(text)

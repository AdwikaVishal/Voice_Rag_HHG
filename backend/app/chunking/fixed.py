"""Strategy 1 — fixed-size chunking with overlap (the baseline/control)."""

from __future__ import annotations

from .base import BaseChunker
from .splitting import split_by_tokens


class FixedSizeChunker(BaseChunker):
    """Split every text unit into fixed-size token chunks with overlap.

    Example configuration::

        FixedSizeChunker(chunk_size=256, overlap=0.20)

    ``overlap`` is a fraction of ``chunk_size``; it is clamped so the stride
    is never zero, which prevents infinite loops. Chunk order is preserved.
    """

    strategy_name = "fixed"

    def split_text(self, text: str) -> list[str]:
        return split_by_tokens(text, self.chunk_size, self.overlap)

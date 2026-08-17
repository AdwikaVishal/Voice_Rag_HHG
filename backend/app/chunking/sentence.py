"""Strategy 2 — sentence-aware (semantic-boundary) chunking.

This is a *sentence-aware* implementation, not an embedding-based semantic
one. Splitting happens at real sentence boundaries and sentences are greedily
combined toward the target chunk size, so coherent thoughts are not broken
mid-sentence. Embedding-based semantic chunking is intentionally deferred to
the retrieval segment.
"""

from __future__ import annotations

from .base import BaseChunker
from .splitting import split_sentences
from .tokenizer import count_tokens


class SentenceChunker(BaseChunker):
    """Greedily combine sentences into chunks of ~``chunk_size`` tokens.

    A single sentence that is longer than ``chunk_size`` is kept whole as its
    own chunk (long sentences are handled safely rather than hard-split).
    """

    strategy_name = "sentence"

    def split_text(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            s_len = count_tokens(sentence)
            if current and current_len + s_len > self.chunk_size:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            if s_len > self.chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_len = 0
                chunks.append(sentence)
            else:
                current.append(sentence)
                current_len += s_len
        if current:
            chunks.append(" ".join(current))
        return [c for c in chunks if c]

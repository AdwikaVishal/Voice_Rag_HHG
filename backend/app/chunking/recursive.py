"""Strategy 3 — recursive / structure-aware chunking.

Splits text using a hierarchy of boundaries::

    paragraph
        ↓
    sentence
        ↓
    token

Adjacent units at a level are greedily merged toward ``chunk_size``; a unit
that is still too large descends to the next level. This mirrors the idea
behind recursive text splitting and suits passages of varying length. No
third-party splitting framework is required.
"""

from __future__ import annotations

from .base import BaseChunker
from .splitting import split_by_tokens, split_paragraphs, split_sentences
from .tokenizer import count_tokens


class RecursiveChunker(BaseChunker):
    """Recursive paragraph -> sentence -> token splitting."""

    strategy_name = "recursive"

    def split_text(self, text: str) -> list[str]:
        paragraphs = split_paragraphs(text)
        if not paragraphs:
            return []
        return self._split_paragraphs(paragraphs)

    # -- paragraph level -------------------------------------------------
    def _split_paragraphs(self, paragraphs: list[str]) -> list[str]:
        result: list[str] = []
        current: list[str] = []
        current_len = 0
        for para in paragraphs:
            p_len = count_tokens(para)
            if current and current_len + p_len > self.chunk_size:
                result.append(" ".join(current))
                current = []
                current_len = 0
            if p_len <= self.chunk_size:
                current.append(para)
                current_len += p_len
            else:
                if current:
                    result.append(" ".join(current))
                    current = []
                    current_len = 0
                result.extend(self._split_sentences(para))
        if current:
            result.append(" ".join(current))
        return [c for c in result if c]

    # -- sentence level --------------------------------------------------
    def _split_sentences(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        result: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            s_len = count_tokens(sentence)
            if current and current_len + s_len > self.chunk_size:
                result.append(" ".join(current))
                current = []
                current_len = 0
            if s_len <= self.chunk_size:
                current.append(sentence)
                current_len += s_len
            else:
                if current:
                    result.append(" ".join(current))
                    current = []
                    current_len = 0
                # token level: the only place an oversized chunk is tolerated
                # is when a single token alone exceeds the target size.
                result.extend(split_by_tokens(sentence, self.chunk_size, 0.0))
        if current:
            result.append(" ".join(current))
        return [c for c in result if c]

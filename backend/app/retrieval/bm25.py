"""BM25 sparse retriever over chunks.

Uses ``rank_bm25.BM25Okapi``. Tokenization reuses the language-neutral
Unicode token approximation (``\\w+``) from the chunking module — the corpus
contains English and Urdu, so ASCII-only tokenization would silently drop
most of the data. Lowercasing is applied for matching; this is safe for
Latin text and a no-op for Arabic-script (Urdu) text, so Indic Unicode is
never corrupted.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from .filters import filter_results
from .models import RetrievalResult, load_chunk_records, make_result
from ..chunking.tokenizer import tokenize

logger = logging.getLogger("retrieval.bm25")

INDEX_FILE = "bm25.pkl"
DEFAULT_EXPANSION = 3


def bm25_tokenize(text: str) -> list[str]:
    """Unicode-safe BM25 tokens (lowercased Unicode word tokens)."""
    return [token.lower() for token in tokenize(text or "")]


class BM25Retriever:
    """BM25Okapi index over the same chunk set used for dense retrieval."""

    def __init__(self, chunks: list[dict[str, Any]], bm25: BM25Okapi | None = None) -> None:
        self.chunks = chunks
        if bm25 is None:
            corpus = [bm25_tokenize(c.get("text", "")) for c in chunks]
            bm25 = BM25Okapi(corpus)
        self.bm25 = bm25

    # -- construction -----------------------------------------------------
    @classmethod
    def build(cls, chunks: list[dict[str, Any]]) -> "BM25Retriever":
        return cls(chunks=chunks)

    @classmethod
    def build_from_chunks(cls, chunk_path: Path, limit: Optional[int] = None) -> "BM25Retriever":
        chunks = list(load_chunk_records(chunk_path))
        if limit:
            chunks = chunks[:limit]
        return cls.build(chunks)

    # -- search -----------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        language: Optional[str] = None,
        expansion: int = DEFAULT_EXPANSION,
    ) -> list[RetrievalResult]:
        """Rank chunks by BM25 score. Empty queries return no results."""
        query_tokens = bm25_tokenize(query)
        if not query_tokens or not self.chunks:
            return []
        k = min(len(self.chunks), top_k * expansion if language else top_k)
        scores = self.bm25.get_scores(query_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for pos in order:
            results.append(make_result(self.chunks[pos], float(scores[pos]), len(results)))
        if language:
            results = filter_results(results, language)
        return results[:top_k]

    # -- persistence ------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": self.chunks, "bm25": self.bm25, "tokenizer": "bm25_tokenize"}
        with (directory / INDEX_FILE).open("wb") as out_file:
            pickle.dump(payload, out_file, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved BM25Retriever -> %s (%d chunks)", directory, len(self.chunks))

    @classmethod
    def load(cls, directory: Path) -> "BM25Retriever":
        directory = Path(directory)
        with (directory / INDEX_FILE).open("rb") as in_file:
            payload = pickle.load(in_file)
        retriever = cls(chunks=payload["chunks"], bm25=payload["bm25"])
        logger.info("Loaded BM25Retriever <- %s (%d chunks)", directory, len(retriever.chunks))
        return retriever

"""Hybrid retrieval — dense (FAISS) + sparse (BM25) + Reciprocal Rank Fusion.

Pipeline for one query::

    query
      ↓
    language detection / filter
      ↓
    dense search        +        BM25 search
                    ↓
           Reciprocal Rank Fusion (k)
                    ↓
                   top_k
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np

from .bm25 import BM25Retriever
from .embeddings import EmbeddingService
from .faiss_store import FAISSStore
from .filters import detect_script_language, filter_results
from .models import RetrievalResult

logger = logging.getLogger("retrieval.hybrid")

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[RetrievalResult]],
    k: int = DEFAULT_RRF_K,
) -> list[RetrievalResult]:
    """Merge ranked result lists with Reciprocal Rank Fusion.

    For each list, position ``r`` (0-indexed) contributes ``1 / (k + r + 1)``
    to its ``chunk_id``; duplicate chunk_ids across lists are summed, so a
    chunk that appears in both dense and BM25 rankings is boosted.
    """
    k = float(k)
    fused: dict[str, float] = {}
    first_seen: dict[str, RetrievalResult] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking):
            fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + 1.0 / (k + rank + 1.0)
            if result.chunk_id not in first_seen:
                first_seen[result.chunk_id] = result

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    merged = []
    for rank, (chunk_id, score) in enumerate(ranked):
        result = first_seen[chunk_id].model_copy(deep=True)
        result.score = score
        result.rank = rank + 1
        result.selected_by_hybrid = True
        merged.append(result)
    return merged


class HybridRetriever:
    """Combines dense + BM25 rankings via RRF, with optional language filter."""

    def __init__(
        self,
        dense: FAISSStore,
        bm25: BM25Retriever,
        embeddings: EmbeddingService,
        rrf_k: float = DEFAULT_RRF_K,
        expansion: int = 3,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.embeddings = embeddings
        self.rrf_k = rrf_k
        self.expansion = expansion

    def _resolve_language(self, query: str, language: Optional[str]) -> Optional[str]:
        if language is None:
            return detect_script_language(query)
        return language

    def search(
        self,
        query: str,
        top_k: int = 10,
        language: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Retrieve ``top_k`` chunks for ``query`` via dense+sparse+RRF."""
        # Short-circuit empty queries to avoid unnecessary embedding/search.
        if not query or not query.strip():
            return []
        query_embedding = self.embeddings.embed_query(query)
        return self.search_embedded(query_embedding, query, top_k=top_k, language=language)

    def search_embedded(
        self,
        query_embedding: np.ndarray,
        query: str,
        top_k: int = 10,
        language: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Like :meth:`search`, but with a pre-computed query embedding."""
        # Short-circuit empty queries regardless of embedding.
        if not query or not query.strip():
            return []
        resolved_language = self._resolve_language(query, language)
        candidates = top_k * self.expansion if resolved_language else top_k

        dense_results = self.dense.search(
            query_embedding, top_k=candidates, language=resolved_language
        )
        bm25_results = self.bm25.search(query, top_k=candidates, language=resolved_language)

        # Filtering already happened inside dense/bm25 when the language was
        # known; applying it again here is a safe no-op and keeps the fused
        # list clean for the "prefer, don't eliminate" guarantee.
        dense_results = filter_results(dense_results, resolved_language)
        bm25_results = filter_results(bm25_results, resolved_language)

        fused = reciprocal_rank_fusion([dense_results, bm25_results], k=self.rrf_k)
        return fused[:top_k]

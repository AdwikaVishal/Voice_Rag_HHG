"""Production retrieval service.

Wraps the benchmark-selected components (``EmbeddingService`` + ``FAISSStore``
+ ``BM25Retriever`` + ``HybridRetriever``) behind a lazy, process-wide
singleton so a request never pays the model-load or index-load cost more than
once.

Components can be injected for tests (a fake encoder plus a small in-memory
index); when nothing is injected the default configuration from
:mod:`app.config` is used and everything loads lazily on the first ``search``
call.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..config import (
    CHUNKING_STRATEGY,
    DEFAULT_TOP_K,
    EMBEDDING_BATCH_SIZE,
    MODEL_NAME,
    RRF_K,
    strategy_index_dir,
)
from .bm25 import BM25Retriever
from .embeddings import EmbeddingService
from .faiss_store import FAISSStore
from .hybrid import HybridRetriever
from .models import RetrievalResult

logger = logging.getLogger("retrieval.production")


class ProductionRetriever:
    """Lazy-loaded hybrid retriever for the selected chunking strategy."""

    def __init__(
        self,
        strategy: Optional[str] = None,
        index_dir: Optional[Path] = None,
        model_name: str = MODEL_NAME,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        rrf_k: float = RRF_K,
        embeddings: Optional[EmbeddingService] = None,
        dense: Optional[FAISSStore] = None,
        bm25: Optional[BM25Retriever] = None,
    ) -> None:
        self.strategy = strategy or CHUNKING_STRATEGY
        self.index_dir = Path(index_dir) if index_dir else strategy_index_dir(self.strategy)
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.rrf_k = float(rrf_k)
        self._embeddings = embeddings
        self._dense = dense
        self._bm25 = bm25
        self._hybrid: Optional[HybridRetriever] = None
        self._chunk_count: Optional[int] = None

    # -- lazy lifecycle --------------------------------------------------
    def _load_dense(self) -> FAISSStore:
        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"No index found at {self.index_dir} (strategy={self.strategy}). "
                "Build it with: python scripts/build_retrieval_index.py "
                f"--strategy {self.strategy}"
            )
        return FAISSStore.load(self.index_dir)

    def _ensure_loaded(self) -> None:
        if self._hybrid is not None:
            return
        dense = self._dense or self._load_dense()
        bm25 = self._bm25 or BM25Retriever.load(self.index_dir)
        embeddings = self._embeddings or EmbeddingService(
            model_name=self.model_name, batch_size=self.batch_size, show_progress=False
        )
        self._dense = dense
        self._bm25 = bm25
        self._embeddings = embeddings
        self._chunk_count = len(dense.chunks)
        self._hybrid = HybridRetriever(
            dense=dense, bm25=bm25, embeddings=embeddings, rrf_k=self.rrf_k
        )
        logger.info(
            "ProductionRetriever ready (strategy=%s, model=%s, chunks=%d)",
            self.strategy,
            self.model_name,
            self._chunk_count,
        )

    @property
    def is_loaded(self) -> bool:
        """Whether the components have been loaded yet."""
        return self._hybrid is not None

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the loaded index."""
        self._ensure_loaded()
        return self._chunk_count

    # -- search ----------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        language: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Retrieve ``top_k`` chunks for ``query`` via the hybrid retriever.

        Empty queries short-circuit to ``[]`` without loading any components.
        ``language`` is optional; when omitted it is auto-detected from the
        query's script.
        """
        query = (query or "").strip()
        if not query or top_k < 1:
            return []
        self._ensure_loaded()
        return self._hybrid.search(query, top_k=top_k, language=language)


@lru_cache(maxsize=1)
def get_production_retriever() -> ProductionRetriever:
    """Process-wide singleton used by the FastAPI layer."""
    return ProductionRetriever()

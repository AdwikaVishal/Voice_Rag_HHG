"""FAISSStore — dense vector index over chunks.

The index type is ``IndexFlatIP`` (exact inner product). Because embeddings
are L2-normalized, inner product equals cosine similarity, giving exact
(non-approximate) results. The corpus is ~100k chunks per strategy, which is
well within reach of an exact, in-memory, single-process index — this also
matches the PRD's preference for an in-process FAISS index (no network hop).

We always keep the position -> chunk mapping so that any result can be
resolved back to ``chunk_id``, ``record_id``, ``query_id``, ``language``,
``text``, ``prev_chunk_id`` and ``next_chunk_id``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import faiss
import numpy as np

# faiss-cpu ships its own libomp.dylib, and torch (loaded first via
# app.retrieval) bundles another. Once both runtimes are live, FAISS's
# parallel inner-product search can fork a region whose worker threads
# deadlock in the join barrier (workers stuck in torch's pool). This only
# triggers when FAISS actually parallelizes — small benchmark indices never
# fork, but the ~100k-chunk production index does. Pin FAISS to a single
# thread so it never forks a parallel region; torch keeps its threads for
# embedding. Safe and idempotent (must run before any search).
faiss.omp_set_num_threads(1)

from .filters import filter_results
from .models import RetrievalResult, load_chunk_records, make_result

logger = logging.getLogger("retrieval.faiss_store")

INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
DEFAULT_EXPANSION = 3


class FAISSStore:
    """In-memory exact FAISS index plus position -> chunk mapping."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        model_name: str = "",
        dim: Optional[int] = None,
        index: Optional[Any] = None,
    ) -> None:
        if index is not None:
            self.index = index
        else:
            matrix = np.asarray(embeddings, dtype=np.float32)
            dim = dim or (matrix.shape[1] if matrix.ndim == 2 else 0)
            self.index = faiss.IndexFlatIP(dim)
            if matrix.ndim == 2 and matrix.shape[0]:
                self.index.add(matrix)
        self.chunks = chunks
        self.model_name = model_name

    # -- construction -----------------------------------------------------
    @classmethod
    def build(cls, chunks: list[dict], embeddings: np.ndarray, model_name: str = "") -> "FAISSStore":
        return cls(chunks=chunks, embeddings=embeddings, model_name=model_name)

    # -- search -----------------------------------------------------------
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        language: Optional[str] = None,
        expansion: int = DEFAULT_EXPANSION,
    ) -> list[RetrievalResult]:
        """Cosine-similarity search over all chunks.

        When ``language`` is a known language, we search ``top_k * expansion``
        candidates and filter down to the requested language so the filter does
        not collapse the candidate pool.
        """
        n = self.index.ntotal
        if n == 0 or top_k < 1:
            return []
        k = min(n, top_k * expansion if language else top_k)
        vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        scores, positions = self.index.search(vector, int(k))
        results = []
        for pos, score in zip(positions[0], scores[0]):
            if pos < 0 or pos >= n:
                continue
            results.append(make_result(self.chunks[int(pos)], float(score), len(results)))
        if language:
            results = filter_results(results, language)
        return results[:top_k]

    # -- persistence ------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        metadata = {
            "model": self.model_name,
            "dim": int(self.index.d),
            "count": int(self.index.ntotal),
            "index_type": "IndexFlatIP",
            "chunks": self.chunks,
        }
        (directory / METADATA_FILE).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Saved FAISSStore -> %s (%d chunks)", directory, self.index.ntotal)

    @classmethod
    def load(cls, directory: Path) -> "FAISSStore":
        directory = Path(directory)
        index = faiss.read_index(str(directory / INDEX_FILE))
        metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
        if int(index.ntotal) != len(metadata.get("chunks", [])):
            raise ValueError(
                f"Corrupt index in {directory}: index has {index.ntotal} vectors "
                f"but metadata lists {len(metadata.get('chunks', []))} chunks"
            )
        store = cls(
            chunks=metadata["chunks"],
            embeddings=np.empty((0, int(index.d)), dtype=np.float32),
            model_name=metadata.get("model", ""),
            index=index,
        )
        logger.info("Loaded FAISSStore <- %s (%d chunks)", directory, index.ntotal)
        return store

    # -- batch helpers ----------------------------------------------------
    @staticmethod
    def build_from_chunks(
        chunk_path: Path,
        embed_fn,
        batch_size: int = 64,
        model_name: str = "",
        limit: Optional[int] = None,
    ) -> "FAISSStore":
        """Load chunks, embed them in batches, and build the store."""
        chunks = list(load_chunk_records(chunk_path))
        if limit:
            chunks = chunks[:limit]
        texts = [c["text"] for c in chunks]
        matrix = embed_in_batches(texts, embed_fn, batch_size)
        if matrix.shape[0] != len(chunks):
            raise ValueError(
                f"embedded {matrix.shape[0]} texts but loaded {len(chunks)} chunks"
            )
        return cls.build(chunks, matrix, model_name=model_name)

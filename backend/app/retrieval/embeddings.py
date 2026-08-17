"""Embedding service — text -> vector.

Uses a small multilingual embedding model
(:data:`DEFAULT_MODEL_NAME`, ``intfloat/multilingual-e5-small``), which:

* supports English + Indic/Arabic scripts,
* is fast enough on CPU,
* is reproducible and loadable locally from the HF cache.

The E5 family requires the ``query: `` / ``passage: `` instruction prefixes;
documents and queries are prefixed accordingly. Embeddings are L2-normalized
so the FAISS index can use cosine similarity via inner product
(``IndexFlatIP``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import torch

# tokenizers' parallel Rust workers can segfault on some Python 3.13/macOS
# setups (SIGSEGV during encode). Disabling parallelism is deterministic and
# prevents the crash; throughput on CPU is unaffected in practice.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Older sentence-transformers (3.x) pulls in ``transformers``, whose lazy
# integration hooks import TensorFlow during ``import sentence_transformers``;
# on macOS TF's shared-library preload deadlocks with "[mutex.cc : 452] RAW:
# Lock blocking ...". We only ever embed with a torch SentenceTransformer, so
# tell transformers to skip TF/JAX before ST is (lazily) imported. Must be set
# before transformers/ST import, which happens on first model load.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

logger = logging.getLogger("retrieval.embeddings")

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_DIMENSION = 384
# Conservative CPU-friendly batch size; the corpus is embedded on a laptop CPU.
DEFAULT_BATCH_SIZE = 32


def configure_cpu_threads() -> None:
    """Pin torch to the available CPU cores.

    The macOS/CPU build of torch defaults to a small thread pool that leaves
    most cores idle during encode; pinning to ``os.cpu_count()`` is a safe,
    deterministic speedup (measured ~5.1 -> ~25 texts/sec on an Apple M2).
    """
    n = max(1, min(32, os.cpu_count() or 1))
    torch.set_num_threads(n)
    torch.set_num_interop_threads(n)


configure_cpu_threads()

_QUERY_PREFIX = "query: "
_DOC_PREFIX = "passage: "


class EmbeddingService:
    """Thin wrapper around a SentenceTransformer embedding model.

    The model is loaded lazily on first use and reused for the lifetime of the
    service (never reloaded per call).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
        encoder: Optional[Any] = None,
        use_query_prefix: bool = True,
        use_doc_prefix: bool = True,
        show_progress: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self._encoder = encoder
        self._model: Optional[Any] = None
        self._load_count = 0
        self.use_query_prefix = bool(use_query_prefix)
        self.use_doc_prefix = bool(use_doc_prefix)
        self.show_progress = bool(show_progress)

    # -- model lifecycle --------------------------------------------------
    def _ensure_loaded(self) -> Any:
        if self._model is None:
            if self._encoder is not None:
                self._model = self._encoder
            else:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s (device=%s)", self.model_name, self.device)
                self._model = SentenceTransformer(self.model_name, device=self.device)
            self._load_count += 1
        return self._model

    @property
    def load_count(self) -> int:
        """Number of times the underlying model was (re)loaded."""
        return self._load_count

    @property
    def dimension(self) -> int:
        """Embedding dimensionality (after loading the model)."""
        model = self._ensure_loaded()
        dim = getattr(model, "get_embedding_dimension", None)
        if not callable(dim):
            dim = getattr(model, "get_sentence_embedding_dimension", None)
        if callable(dim):
            return int(dim())
        return DEFAULT_DIMENSION

    # -- encode -----------------------------------------------------------
    def _encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        model = self._ensure_loaded()
        return np.asarray(
            model.encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=bool(show_progress),
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def embed_documents(self, texts: Iterable[str], show_progress: Optional[bool] = None) -> np.ndarray:
        """Embed chunk/passage texts. Applies the ``passage: `` prefix.

        ``show_progress`` defaults to the service-level flag; pass an explicit
        value to override it for a single call (bulk indexing wants a progress
        bar, small unit-test batches do not).
        """
        if show_progress is None:
            show_progress = self.show_progress
        prefixed = [self._prefix(text, self.use_doc_prefix, _DOC_PREFIX) for text in texts]
        return self._encode(prefixed, show_progress=show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query. Applies the ``query: `` prefix."""
        text = self._prefix(query, self.use_query_prefix, _QUERY_PREFIX)
        vectors = self._encode([text], show_progress=False)
        return vectors[0]

    @staticmethod
    def _prefix(text: str, enabled: bool, prefix: str) -> str:
        text = (text or "").strip()
        if enabled and text and not text.startswith(prefix):
            return prefix + text
        return text


def embed_callback_factory(service: EmbeddingService) -> Callable[[Sequence[str]], np.ndarray]:
    """Return a callable ``chunks_texts -> matrix`` bound to ``service``."""
    return service.embed_documents


def embed_in_batches(
    texts: Sequence[str],
    embed_fn: Callable[[Sequence[str]], np.ndarray],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Embed ``texts`` in fixed-size batches into a single float32 matrix.

    Progress bars are suppressed per batch (the caller decides whether to show
    a bar over the whole job).
    """
    batch_size = max(1, int(batch_size))
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.append(np.asarray(embed_fn(batch, show_progress=False), dtype=np.float32))
    if not vectors:
        return np.empty((0, DEFAULT_DIMENSION), dtype=np.float32)
    return np.concatenate(vectors, axis=0)

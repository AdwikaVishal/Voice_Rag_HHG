"""Retrieval components: embeddings, FAISS dense store, BM25, filters, hybrid.

Build a full retrieval stack for a chunking strategy with::

    from app.retrieval import (
        EmbeddingService,
        FAISSStore,
        BM25Retriever,
        HybridRetriever,
    )

    embeddings = EmbeddingService()
    dense = FAISSStore.load(indexes_dir / "fixed")
    bm25 = BM25Retriever.load(indexes_dir / "fixed")
    hybrid = HybridRetriever(dense=dense, bm25=bm25, embeddings=embeddings)
    results = hybrid.search("query text", top_k=10)
"""

from __future__ import annotations

import os

# torch, faiss-cpu and scikit-learn each ship their own libomp.dylib, so up to
# three OpenMP runtimes end up in one process. On macOS the second runtime
# aborts the process on first parallel use ("OMP: Error #15"). Python cannot
# force a single runtime, so we accept the documented workaround before any
# OpenMP-using library is imported / used. This makes the safe configuration
# (a real torch op preceding faiss use) deterministic for tests, the API and
# the benchmark scripts alike.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# torch, faiss-cpu and scikit-learn each ship their own libomp.dylib, so up to
# three OpenMP runtimes end up in one process. On macOS the second runtime
# aborts the process on first parallel use ("OMP: Error #15"). Python cannot
# force a single runtime, so we accept the documented workaround before any
# OpenMP-using library is imported / used. This makes the safe configuration
# (a real torch op preceding faiss use) deterministic for tests, the API and
# the benchmark scripts alike.
#
# We ALSO pin OpenMP to a single thread process-wide. Two independent libomp
# runtimes (torch's + faiss-cpu's) can deadlock when FAISS forks a parallel
# region: its worker threads end up stuck in torch's pool and the join never
# completes. This only triggers once FAISS actually parallelizes, i.e. on the
# ~100k-chunk production index (small benchmark indices never fork), and only
# on threads that did not see faiss.omp_set_num_threads(1) — the FastAPI
# /search handler runs on a worker thread, so a per-thread setting is not
# enough. Setting OMP_NUM_THREADS before either runtime initializes makes the
# whole process single-threaded for OpenMP work and removes the deadlock.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Import torch BEFORE faiss-cpu. On macOS + Python 3.13, importing faiss
# first and then initializing torch (via sentence-transformers) segfaults
# inside a threading wait due to a conflict between their bundled BLAS /
# thread pools. Importing torch first makes the order safe and deterministic.
import torch  # noqa: F401

# In addition, faiss_store pins FAISS to a single OpenMP thread
# (faiss.omp_set_num_threads(1)): with torch + faiss-cpu both loaded, FAISS's
# parallel knn over the large production index deadlocks in a libomp join
# barrier. See the comment in faiss_store.py.

from .bm25 import BM25Retriever
from .embeddings import DEFAULT_MODEL_NAME, EmbeddingService
from .evaluation import (
    aggregate_recall,
    build_eval_queries,
    load_eval_set,
    percentiles,
    recall_at_k,
    relevant_chunk_ids,
    save_eval_set,
    source_passage_chunk_ids,
    summary_stats,
)
from .faiss_store import FAISSStore
from .filters import detect_script_language, filter_results, known_language, match_language
from .hybrid import DEFAULT_RRF_K, HybridRetriever, reciprocal_rank_fusion
from .models import RetrievalResult, load_chunk_records
from .production import ProductionRetriever, get_production_retriever
from .sampling import (
    build_manifest,
    choose_record_sample_size,
    chunks_for_records,
    count_chunks,
    record_list_from_corpus,
    sample_record_ids,
)

__all__ = [
    "EmbeddingService",
    "FAISSStore",
    "BM25Retriever",
    "HybridRetriever",
    "RetrievalResult",
    "reciprocal_rank_fusion",
    "detect_script_language",
    "filter_results",
    "known_language",
    "match_language",
    "load_chunk_records",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_RRF_K",
    # production
    "ProductionRetriever",
    "get_production_retriever",
    # benchmark sampling
    "record_list_from_corpus",
    "count_chunks",
    "choose_record_sample_size",
    "sample_record_ids",
    "chunks_for_records",
    "build_manifest",
    # evaluation
    "build_eval_queries",
    "save_eval_set",
    "load_eval_set",
    "relevant_chunk_ids",
    "source_passage_chunk_ids",
    "recall_at_k",
    "aggregate_recall",
    "percentiles",
    "summary_stats",
]

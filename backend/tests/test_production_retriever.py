"""Tests for the production retrieval service, config, and API layer.

These run entirely on a fake encoder and a small in-memory / temp-dir index —
the real SentenceTransformer model is never loaded (see
``tests/test_embedding_integration.py`` for explicit integration tests).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import CHUNKS, FakeEncoder, build_small_stack  # noqa: E402

from app.config import CHUNKING_STRATEGY, strategy_index_dir  # noqa: E402
from app.retrieval import (  # noqa: E402
    BM25Retriever,
    EmbeddingService,
    FAISSStore,
    ProductionRetriever,
)
from app.retrieval.embeddings import embed_in_batches  # noqa: E402
from app.retrieval.models import RetrievalResult  # noqa: E402
from app.retrieval.production import get_production_retriever  # noqa: E402


def build_temp_index(tmp: Path, chunks=None):
    """Build a FAISS + BM25 index into ``tmp`` using the fake encoder."""
    chunks = chunks or CHUNKS
    svc = EmbeddingService(encoder=FakeEncoder(32), use_query_prefix=True, use_doc_prefix=True)
    matrix = embed_in_batches([c["text"] for c in chunks], svc.embed_documents, 4)
    dense = FAISSStore.build(chunks, matrix)
    dense.save(tmp)
    BM25Retriever.build(chunks).save(tmp)
    return svc


class TestConfig(unittest.TestCase):
    def test_default_strategy_is_recursive(self):
        self.assertEqual(CHUNKING_STRATEGY, "recursive")

    def test_strategy_index_dir_nested(self):
        # Production default points at the full-corpus index (data/indexes/<strategy>);
        # the Segment 3 benchmark sample lives under data/indexes/benchmark/.
        expected = Path(BASE_DIR) / "data" / "indexes" / "recursive"
        self.assertEqual(strategy_index_dir("recursive"), expected)


class TestProductionRetriever(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.index_dir = Path(self.tmp.name)
        self.svc = build_temp_index(self.index_dir)
        self.retriever = ProductionRetriever(
            strategy="recursive",
            index_dir=self.index_dir,
            embeddings=self.svc,
            model_name="fake-model",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_lazy_until_first_search(self):
        self.assertFalse(self.retriever.is_loaded)
        self.retriever.search("the capital of France", top_k=3)
        self.assertTrue(self.retriever.is_loaded)

    def test_empty_query_short_circuits_without_loading(self):
        self.assertEqual(self.retriever.search("   ", top_k=5), [])
        self.assertEqual(self.retriever.search("", top_k=5), [])
        self.assertFalse(self.retriever.is_loaded)

    def test_top_k_respected(self):
        results = self.retriever.search("the capital of France is Paris", top_k=2)
        self.assertEqual(len(results), 2)

    def test_english_query_metadata(self):
        results = self.retriever.search("where is the capital of France?", top_k=3)
        self.assertTrue(results)
        r = results[0]
        self.assertEqual(r.metadata["chunk_id"], r.chunk_id)
        for field in ("record_id", "query_id", "passage_index", "language", "prev_chunk_id", "next_chunk_id"):
            self.assertIn(field, r.metadata)
        self.assertEqual(r.metadata["language"], "eng_Latn")

    def test_urdu_query_returns_urdu_chunks(self):
        results = self.retriever.search("کارپوریشن ایک کمپنی ہے؟", top_k=3)
        self.assertTrue(results)
        self.assertTrue(all(r.metadata["language"] == "urd_Arab" for r in results))

    def test_language_filter_preferred(self):
        results = self.retriever.search("the capital of France", top_k=3, language="eng_Latn")
        self.assertTrue(results)
        self.assertTrue(all(r.metadata["language"] == "eng_Latn" for r in results))

    def test_loads_index_from_disk(self):
        # Non-injected dense/bm25: the service must load them from the directory.
        fresh = ProductionRetriever(strategy="recursive", index_dir=self.index_dir, embeddings=self.svc)
        results = fresh.search("Bears hibernate during the winter months", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(fresh.chunk_count, len(CHUNKS))
        self.assertEqual(fresh.strategy, "recursive")

    def test_missing_index_raises(self):
        missing = ProductionRetriever(strategy="nope", index_dir=Path(self.tmp.name) / "missing")
        with self.assertRaises(FileNotFoundError):
            missing.search("anything", top_k=3)
        self.assertFalse(missing.is_loaded)

    def test_strategy_default(self):
        r = ProductionRetriever()
        self.assertEqual(r.strategy, CHUNKING_STRATEGY)


class TestSingleton(unittest.TestCase):
    def test_get_production_retriever_is_cached(self):
        get_production_retriever.cache_clear()
        a = get_production_retriever()
        b = get_production_retriever()
        self.assertIs(a, b)


class FakeRetriever:
    """Stand-in for the production retriever used by the API tests."""

    strategy = "recursive"

    def __init__(self):
        self.index_dir = Path("/fake")
        self.model_name = "fake-model"
        self.rrf_k = 60.0
        self.is_loaded = True
        self.chunk_count = 5

    def search(self, query, top_k=10, language=None):
        return [
            RetrievalResult(
                chunk_id=f"chunk{i}",
                score=round(1.0 / (i + 1), 5),
                text=f"result {i} for {query}",
                rank=i + 1,
                metadata={
                    "chunk_id": f"chunk{i}",
                    "record_id": f"rec{i}",
                    "query_id": 1,
                    "passage_index": i,
                    "language": language or "eng_Latn",
                    "prev_chunk_id": None,
                    "next_chunk_id": None,
                },
            )
            for i in range(min(top_k, 2))
        ]


class TestSearchAPI(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from app.main import app

        self.client = TestClient(app)
        self.patcher = mock.patch("app.main.get_production_retriever", return_value=FakeRetriever())
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_root(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["message"], "Voice RAG API is running")

    def test_retriever_info(self):
        resp = self.client.get("/retriever/info")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["strategy"], "recursive")
        self.assertEqual(resp.json()["chunks"], 5)

    def test_search_returns_results(self):
        resp = self.client.post("/search", json={"query": "the capital of France?", "top_k": 3})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["query"], "the capital of France?")
        self.assertEqual(body["strategy"], "recursive")
        self.assertEqual(len(body["results"]), 2)
        self.assertEqual(body["detected_language"], "eng_Latn")

    def test_search_urdu_detects_language(self):
        resp = self.client.post("/search", json={"query": "کارپوریشن کیا ہے؟"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["detected_language"], "urd_Arab")

    def test_search_empty_query_rejected(self):
        resp = self.client.post("/search", json={"query": "   "})
        self.assertEqual(resp.status_code, 422)

    def test_search_top_k_bounds(self):
        resp = self.client.post("/search", json={"query": "q", "top_k": 0})
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()

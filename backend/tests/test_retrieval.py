"""Unit tests for the retrieval module (Segment 3).

Run from the backend/ directory:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.retrieval import EmbeddingService, FAISSStore, HybridRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import embed_in_batches
from app.retrieval.evaluation import (
    aggregate_recall,
    build_eval_queries,
    percentile,
    percentiles,
    recall_at_k,
    relevant_chunk_ids,
    save_eval_set,
    source_passage_chunk_ids,
)
from app.retrieval.filters import detect_script_language, filter_results, match_language
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.models import RetrievalResult, load_chunk_records
from app.retrieval.sampling import (
    choose_record_sample_size,
    chunks_for_records,
    sample_record_ids,
)

# Shared fixtures live in tests/support.py (discover adds the tests dir to
# sys.path, so this is a top-level import).
import sys as _sys

if str(Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import CHUNKS, FakeEncoder, QUERIES, build_small_stack, make_chunk  # noqa: E402


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


class TestEmbeddingService(unittest.TestCase):
    def test_fake_encoder_used_once(self):
        encoder = FakeEncoder()
        svc = EmbeddingService(encoder=encoder, use_doc_prefix=False)
        svc.embed_documents(["a", "b", "c"])
        svc.embed_documents(["d"])
        svc.embed_query("q")
        self.assertEqual(encoder.encode_calls, 3)

    def test_show_progress_flagged_for_documents(self):
        # The real sentence-transformers encode accepts show_progress_bar; our
        # service must forward the service-level flag for document embedding.
        encoder = FakeEncoder()
        encoder.encode_kwargs = []
        def capture(texts, **kwargs):
            encoder.encode_calls += 1
            encoder.encode_kwargs.append(kwargs)
            return FakeEncoder.encode(encoder, texts, **kwargs)
        encoder.encode = capture
        svc = EmbeddingService(encoder=encoder, show_progress=True)
        svc.embed_documents(["a"])
        self.assertTrue(encoder.encode_kwargs[0]["show_progress_bar"])

    def test_query_never_shows_progress(self):
        encoder = FakeEncoder()
        encoder.encode_kwargs = []
        def capture(texts, **kwargs):
            encoder.encode_calls += 1
            encoder.encode_kwargs.append(kwargs)
            return FakeEncoder.encode(encoder, texts, **kwargs)
        encoder.encode = capture
        svc = EmbeddingService(encoder=encoder, show_progress=True)
        svc.embed_query("q")
        self.assertFalse(encoder.encode_kwargs[0]["show_progress_bar"])


# --------------------------------------------------------------------------
# FAISS
# --------------------------------------------------------------------------


class TestFAISSStore(unittest.TestCase):
    def setUp(self):
        # Prefix-free service so embedding a chunk's own text yields an exact
        # vector match (cosine = 1) regardless of the deterministic fake encoder.
        self.svc = EmbeddingService(
            encoder=FakeEncoder(32),
            use_query_prefix=False,
            use_doc_prefix=False,
        )
        self.chunks = CHUNKS
        matrix = embed_in_batches([c["text"] for c in self.chunks], self.svc.embed_documents, 4)
        self.dense = FAISSStore.build(self.chunks, matrix)

    def test_build_and_search(self):
        # Exact-match query: the top hit must be the chunk whose own text is the query.
        q = self.svc.embed_query(CHUNKS[0]["text"])  # "The capital of France is Paris..."
        results = self.dense.search(q, top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].chunk_id, "c1")
        self.assertAlmostEqual(results[0].score, 1.0, places=4)  # cosine ~1 for exact match

    def test_search_respects_top_k(self):
        q = self.svc.embed_query(CHUNKS[2]["text"])  # "Bears hibernate..."
        for k in (1, 2, 5):
            self.assertEqual(len(self.dense.search(q, top_k=k)), min(k, len(self.chunks)))
            self.assertEqual(self.dense.search(q, top_k=k)[0].chunk_id, "c3")

    def test_metadata_mapping(self):
        q = self.svc.embed_query(CHUNKS[0]["text"])
        result = self.dense.search(q, top_k=1)[0]
        self.assertEqual(result.metadata["chunk_id"], result.chunk_id)
        self.assertIn("record_id", result.metadata)
        self.assertIn("text", result.metadata)
        self.assertEqual(result.metadata["language"], "eng_Latn")

    def test_save_load_roundtrip(self):
        q = self.svc.embed_query(CHUNKS[0]["text"])
        expected = [r.chunk_id for r in self.dense.search(q, top_k=5)]
        with tempfile.TemporaryDirectory() as tmp:
            self.dense.save(Path(tmp))
            loaded = FAISSStore.load(Path(tmp))
            self.assertEqual(loaded.index.ntotal, self.dense.index.ntotal)
            actual = [r.chunk_id for r in loaded.search(q, top_k=5)]
            self.assertEqual(actual, expected)

    def test_empty_store_search(self):
        dense = FAISSStore.build([], np.empty((0, 32), dtype=np.float32))
        self.assertEqual(dense.search(np.zeros(32, dtype=np.float32), top_k=5), [])


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


class TestBM25(unittest.TestCase):
    def test_build_and_search(self):
        bm25 = BM25Retriever.build(CHUNKS)
        results = bm25.search("capital of France Paris", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertIn(results[0].chunk_id, {"c1", "c2"})
        self.assertGreater(results[0].score, 0.0)

    def test_urdu_search(self):
        bm25 = BM25Retriever.build(CHUNKS)
        results = bm25.search("کارپوریشن کمپنی", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertIn("c5", [r.chunk_id for r in results])

    def test_empty_query(self):
        bm25 = BM25Retriever.build(CHUNKS)
        self.assertEqual(bm25.search("", top_k=5), [])
        self.assertEqual(bm25.search("   ", top_k=5), [])

    def test_save_load(self):
        bm25 = BM25Retriever.build(CHUNKS)
        with tempfile.TemporaryDirectory() as tmp:
            bm25.save(Path(tmp))
            loaded = BM25Retriever.load(Path(tmp))
            self.assertEqual(len(loaded.chunks), len(CHUNKS))
            r = loaded.search("capital of France", top_k=1)[0]
            self.assertIn(r.chunk_id, {"c1", "c2"})


# --------------------------------------------------------------------------
# RRF
# --------------------------------------------------------------------------


def _result(chunk_id, score, rank):
    chunk = make_chunk(chunk_id, "text " + chunk_id)
    return RetrievalResult(chunk_id=chunk_id, score=score, text="text", metadata=chunk, rank=rank)


class TestRRF(unittest.TestCase):
    def test_duplicates_merge_and_sum(self):
        ranking_a = [_result("x", 1.0, 1), _result("y", 1.0, 2)]
        ranking_b = [_result("x", 2.0, 1), _result("z", 2.0, 2)]
        fused = reciprocal_rank_fusion([ranking_a, ranking_b], k=60)
        ids = [r.chunk_id for r in fused]
        self.assertEqual(len(ids), 3)
        self.assertEqual(set(ids), {"x", "y", "z"})
        x = next(r for r in fused if r.chunk_id == "x")
        # present in both rankings at rank 1 -> 1/61 + 1/61
        self.assertAlmostEqual(x.score, 2.0 / 61.0, places=5)

    def test_configurable_k(self):
        ranking_a = [_result("x", 1.0, 1), _result("y", 1.0, 2)]
        fused_high = reciprocal_rank_fusion([ranking_a], k=100)
        fused_low = reciprocal_rank_fusion([ranking_a], k=10)
        self.assertAlmostEqual(fused_high[0].score, 1.0 / 101.0, places=6)
        self.assertAlmostEqual(fused_low[0].score, 1.0 / 11.0, places=6)

    def test_order_preserved_by_score(self):
        ranking = [_result("a", 1.0, 1), _result("b", 1.0, 2), _result("c", 1.0, 3)]
        fused = reciprocal_rank_fusion([ranking], k=60)
        self.assertEqual([r.chunk_id for r in fused], ["a", "b", "c"])


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------


class TestFilters(unittest.TestCase):
    def test_script_detection(self):
        self.assertEqual(detect_script_language("what is a corporation?"), "eng_Latn")
        self.assertEqual(detect_script_language("کارپوریشن کیا ہے؟"), "urd_Arab")
        self.assertIsNone(detect_script_language(""))

    def test_filter_by_language(self):
        results = [
            RetrievalResult(chunk_id="a", score=1.0, text="t", metadata=make_chunk("a", "t", language="eng_Latn")),
            RetrievalResult(chunk_id="b", score=1.0, text="t", metadata=make_chunk("b", "t", language="urd_Arab")),
        ]
        filtered = filter_results(results, "urd_Arab")
        self.assertEqual([r.chunk_id for r in filtered], ["b"])

    def test_unknown_language_does_not_eliminate(self):
        results = [
            RetrievalResult(chunk_id="a", score=1.0, text="t", metadata=make_chunk("a", "t", language="eng_Latn")),
        ]
        self.assertEqual(len(filter_results(results, "zzz_XXX")), 1)
        self.assertEqual(len(filter_results(results, None)), 1)

    def test_match_language_fallback(self):
        chunk = make_chunk("a", "t", language="")
        self.assertTrue(match_language(chunk, "urd_Arab"))  # target_lang match


# --------------------------------------------------------------------------
# Hybrid
# --------------------------------------------------------------------------


class TestHybrid(unittest.TestCase):
    def test_returns_ranked_results_with_metadata(self):
        svc, dense, bm25 = build_small_stack()
        hybrid = HybridRetriever(dense=dense, bm25=bm25, embeddings=svc)
        results = hybrid.search(QUERIES[0], top_k=3)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn(r.chunk_id, {c["chunk_id"] for c in CHUNKS})
            self.assertIn("text", r.metadata)
            self.assertEqual(r.metadata["chunk_id"], r.chunk_id)

    def test_top_k_respected(self):
        svc, dense, bm25 = build_small_stack()
        hybrid = HybridRetriever(dense=dense, bm25=bm25, embeddings=svc)
        self.assertEqual(len(hybrid.search(QUERIES[0], top_k=1)), 1)
        self.assertEqual(len(hybrid.search(QUERIES[0], top_k=4)), 4)

    def test_language_filter_in_hybrid(self):
        svc, dense, bm25 = build_small_stack()
        hybrid = HybridRetriever(dense=dense, bm25=bm25, embeddings=svc)
        results = hybrid.search("کارپوریشن کمپنی", top_k=5, language="urd_Arab")
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.metadata["language"], "urd_Arab")

    def test_hybrid_uses_both_sources(self):
        svc, dense, bm25 = build_small_stack()
        hybrid = HybridRetriever(dense=dense, bm25=bm25, embeddings=svc)
        q = svc.embed_query(QUERIES[0])
        dense_only = [r.chunk_id for r in dense.search(q, top_k=5)]
        bm25_only = [r.chunk_id for r in bm25.search(QUERIES[0], top_k=5)]
        hybrid_ids = [r.chunk_id for r in hybrid.search_embedded(q, QUERIES[0], top_k=5)]
        union = set(dense_only) | set(bm25_only)
        # hybrid is a re-ranking of the union, so it must not contain anything
        # that appeared in neither source ranking
        self.assertTrue(set(hybrid_ids) <= union)
        self.assertGreaterEqual(len(hybrid_ids), 1)


# --------------------------------------------------------------------------
# Chunk loading
# --------------------------------------------------------------------------


class TestChunkLoading(unittest.TestCase):
    def test_load_chunk_records(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write('{"chunk_id": "x1", "text": "hello"}\n{"chunk_id": "x2", "text": "world"}\n')
            path = Path(f.name)
        records = list(load_chunk_records(path))
        path.unlink()
        self.assertEqual([r["chunk_id"] for r in records], ["x1", "x2"])


# --------------------------------------------------------------------------
# Sampling (deterministic, strategy-fair)
# --------------------------------------------------------------------------


class TestSampling(unittest.TestCase):
    RECORDS = [f"rec_{i:03d}" for i in range(100)]

    def test_seeded_sample_is_deterministic(self):
        a = sample_record_ids(self.RECORDS, k=20, seed=42)
        b = sample_record_ids(self.RECORDS, k=20, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 20)

    def test_different_seed_changes_sample(self):
        a = sample_record_ids(self.RECORDS, k=20, seed=42)
        b = sample_record_ids(self.RECORDS, k=20, seed=7)
        self.assertNotEqual(a, b)

    def test_sample_has_no_duplicates_and_respects_k(self):
        for k in (1, 50, 200):
            self.assertEqual(len(set(sample_record_ids(self.RECORDS, k=k))), min(k, 100))

    def test_choose_record_sample_size_converts_chunk_target(self):
        # 10k chunks out of 100k total over 5k records -> ~500 records
        self.assertEqual(choose_record_sample_size(10000, 5000, 100000), 500)
        self.assertEqual(choose_record_sample_size(10000, 5000, 100918), 495)
        self.assertLessEqual(choose_record_sample_size(10**9, 5000, 100000), 5000)
        self.assertGreaterEqual(choose_record_sample_size(1, 5000, 100000), 1)

    def test_chunks_for_records_filters(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(
                '{"chunk_id": "c1", "record_id": "r1", "text": "a"}\n'
                '{"chunk_id": "c2", "record_id": "r2", "text": "b"}\n'
                '{"chunk_id": "c3", "record_id": "r1", "text": "c"}\n'
            )
            path = Path(f.name)
        ids = [c["chunk_id"] for c in chunks_for_records(path, ["r1"])]
        path.unlink()
        self.assertEqual(ids, ["c1", "c3"])


# --------------------------------------------------------------------------
# Evaluation helpers (Recall@K, percentiles, ground truth)
# --------------------------------------------------------------------------


class TestEvaluationMetrics(unittest.TestCase):
    def test_recall_at_k_binary(self):
        relevant = {"c1", "c5"}
        self.assertEqual(recall_at_k(["c1", "c2"], relevant, 5), 1.0)
        self.assertEqual(recall_at_k(["c2", "c3"], relevant, 5), 0.0)
        self.assertEqual(recall_at_k(["c2", "c5"], relevant, 1), 0.0)  # only top-1 counted
        self.assertEqual(recall_at_k(["c2"], relevant, 0), 0.0)

    def test_aggregate_recall(self):
        hits = [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
        out = aggregate_recall(hits, (1, 3, 5, 10))
        self.assertEqual(out["recall@1"], 0.5)
        self.assertEqual(out["recall@3"], 0.5)
        self.assertEqual(out["recall@5"], 1.0)
        self.assertEqual(out["recall@10"], 1.0)

    def test_percentile_nearest_rank(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2)  # nearest-rank p50
        self.assertEqual(percentile([5], 100), 5)
        self.assertEqual(percentiles([1, 2, 3, 4])["p100"], 4)
        self.assertEqual(percentiles([1, 2, 3, 4])["p50"], 2)
        self.assertEqual(percentiles([1, 2, 3, 4])["p70"], 3)
        with self.assertRaises(ValueError):
            percentile([], 50)

    def test_build_eval_queries_uses_only_selected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(
                '{"record_id": "r1", "query_id": 1, "query": "q1", '
                '"passages": [{"passage_index": 0, "is_selected": 1}, {"passage_index": 1, "is_selected": 0}]}\n'
                '{"record_id": "r2", "query_id": 2, "query": "q2", '
                '"passages": [{"passage_index": 0, "is_selected": 0}]}\n'
                '{"record_id": "r3", "query_id": 3, "query": "q3", '
                '"passages": [{"passage_index": 0, "is_selected": 1}]}\n'
            )
            path = Path(f.name)
        queries = build_eval_queries(path, ["r1", "r3", "r9"], max_queries=10)
        path.unlink()
        self.assertEqual(len(queries), 2)
        self.assertEqual({q["record_id"] for q in queries}, {"r1", "r3"})
        self.assertEqual(queries[0]["relevant_passage_indices"], [0])

    def test_relevant_chunk_ids_maps_selected_passages_to_chunks(self):
        queries = [
            {
                "query_id": 1,
                "record_id": "r1",
                "relevant_passage_indices": [0, 2],
            }
        ]
        chunks = [
            {"chunk_id": "c_eng", "record_id": "r1", "passage_index": 0},
            {"chunk_id": "c_urd", "record_id": "r1", "passage_index": 0},
            {"chunk_id": "c_other", "record_id": "r1", "passage_index": 1},
            {"chunk_id": "c_p2", "record_id": "r1", "passage_index": 2},
            {"chunk_id": "c_r2", "record_id": "r2", "passage_index": 0},
        ]
        mapping = relevant_chunk_ids(queries, chunks)
        self.assertEqual(mapping[1], {"c_eng", "c_urd", "c_p2"})

    def test_source_passage_chunk_ids(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(
                '{"chunk_id": "c1", "record_id": "r1", "passage_index": 0}\n'
                '{"chunk_id": "c2", "record_id": "r1", "passage_index": 1}\n'
                '{"chunk_id": "c3", "record_id": "r1", "passage_index": 0}\n'
            )
            path = Path(f.name)
        ids = source_passage_chunk_ids(path, "r1", 0)
        path.unlink()
        self.assertEqual(sorted(ids), ["c1", "c3"])


if __name__ == "__main__":
    unittest.main()

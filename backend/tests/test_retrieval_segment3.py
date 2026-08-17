"""Segment 3 — end-to-end retrieval behaviour tests (groups A–E).

These run entirely on the deterministic fake encoder and a small in-memory
index; the real SentenceTransformer model is never loaded here (see
``tests/test_embedding_integration.py`` for explicit integration tests).

Groups covered:
  A  Known-answerable queries (CDG, English + Urdu) must retrieve evidence.
  B  Missing-evidence queries must not fabricate evidence — the system
     distinguishes "retrieval returned nothing useful" from "corpus coverage
     gap", and still degrades gracefully.
  C  Language filtering: eng_Latn / urd_Arab filters are applied by every
     retriever and by auto-detection.
  D  Out-of-domain noise queries: graceful degradation, top_k respected, no
     crash, metadata always present.
  E  Retrieval component correctness: top_k bounds, score ordering, empty and
     malformed queries, and the hybrid-selection provenance flag.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import CHUNKS, FakeEncoder, build_small_stack, make_chunk  # noqa: E402

from app.retrieval import EmbeddingService, FAISSStore, HybridRetriever  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402
from app.retrieval.filters import detect_script_language  # noqa: E402


def _svc():
    return EmbeddingService(encoder=FakeEncoder(32), use_query_prefix=True, use_doc_prefix=True)


def build_stack(chunks):
    svc = _svc()
    dense = FAISSStore.build(chunks, svc.embed_documents([c["text"] for c in chunks]))
    bm25 = BM25Retriever.build(chunks)
    return svc, dense, bm25


# --------------------------------------------------------------------------
# Fixtures: a small bilingual corpus with real evidence for CDG and nothing
# answering "capital of France" or "invented the telephone".
# --------------------------------------------------------------------------

CDG_CHUNKS = [
    make_chunk(
        "cdg_eng",
        "Which airport is CDG? CDG is officially named Roissy Charles de Gaulle "
        "Airport and it serves Paris.",
        record_id="msmarco_xi_000917",
        passage_index=3,
    ),
    make_chunk(
        "cdg_urd",
        "سی ڈی جی ہوائی اڈا کیا ہے؟ سی ڈی جی کا سرکاری نام روسی شارل ڈی گال ہوائی "
        "اڈا ہے جو پیریس کی خدمت کرتا ہے۔",
        language="urd_Arab",
        record_id="msmarco_xi_000917",
        passage_index=3,
    ),
    make_chunk("fr_hist", "Louis XIV moved the seat of French government to Versailles."),
    make_chunk("bear", "Bears hibernate during the winter months."),
]

EN_CDG_QUERY = "What is CDG airport?"
UR_CDG_QUERY = "سی ڈی جی ہوائی اڈا کیا ہے؟"


class TestGroupA_KnownEvidenceRetrieved(unittest.TestCase):
    def setUp(self):
        self.svc, self.dense, self.bm25 = build_stack(CDG_CHUNKS)
        self.hybrid = HybridRetriever(dense=self.dense, bm25=self.bm25, embeddings=self.svc)

    def test_english_cdg_evidence_in_hybrid_top_k(self):
        results = self.hybrid.search(EN_CDG_QUERY, top_k=5)
        self.assertIn("cdg_eng", [r.chunk_id for r in results])

    def test_urdu_cdg_evidence_in_hybrid_top_k(self):
        results = self.hybrid.search(UR_CDG_QUERY, top_k=5)
        ids = [r.chunk_id for r in results]
        self.assertIn("cdg_urd", ids)
        self.assertNotIn("cdg_eng", ids)  # Urdu query is filtered to Urdu chunks

    def test_english_cdg_evidence_in_each_source(self):
        qvec = self.svc.embed_query(EN_CDG_QUERY)
        self.assertIn("cdg_eng", [r.chunk_id for r in self.bm25.search(EN_CDG_QUERY, top_k=5)])
        self.assertIn("cdg_eng", [r.chunk_id for r in self.dense.search(qvec, top_k=5)])


class TestGroupB_MissingEvidenceNotFabricated(unittest.TestCase):
    def setUp(self):
        self.svc, self.dense, self.bm25 = build_stack(CDG_CHUNKS)
        self.hybrid = HybridRetriever(dense=self.dense, bm25=self.bm25, embeddings=self.svc)

    def test_capital_of_france_has_no_grounding_chunk(self):
        # Corpus coverage check: the evidence a downstream answerer would need
        # simply does not exist here. This is a coverage failure, not a
        # retrieval bug — the index must never fabricate it.
        corpus_text = " ".join(c["text"] for c in CDG_CHUNKS).lower()
        for phrase in ("paris is the capital", "capital of france is paris"):
            self.assertNotIn(phrase, corpus_text)

    def test_invented_telephone_has_no_grounding_chunk(self):
        corpus_text = " ".join(c["text"] for c in CDG_CHUNKS).lower()
        for phrase in ("invented the telephone", "graham bell", "bell"):
            self.assertNotIn(phrase, corpus_text)

    def test_no_evidence_returns_results_gracefully(self):
        for q in ("What is the capital of France?", "Who invented the telephone?"):
            results = self.hybrid.search(q, top_k=5)
            self.assertLessEqual(len(results), 5)  # never more than requested
            self.assertTrue(isinstance(results, list))
            for r in results:
                self.assertIn("text", r.metadata)  # results are real corpus chunks
            # None of the returned chunks can contain the answer statement.
            for r in results:
                self.assertNotIn("paris is the capital", r.text.lower())
                self.assertNotIn("invented the telephone", r.text.lower())


class TestGroupC_LanguageFiltering(unittest.TestCase):
    def setUp(self):
        self.svc, self.dense, self.bm25 = build_stack(CDG_CHUNKS)
        self.hybrid = HybridRetriever(dense=self.dense, bm25=self.bm25, embeddings=self.svc)

    def test_script_detection(self):
        self.assertEqual(detect_script_language(EN_CDG_QUERY), "eng_Latn")
        self.assertEqual(detect_script_language(UR_CDG_QUERY), "urd_Arab")

    def test_dense_language_filter(self):
        qvec = self.svc.embed_query(UR_CDG_QUERY)
        results = self.dense.search(qvec, top_k=5, language="urd_Arab")
        self.assertTrue(results)
        self.assertTrue(all(r.metadata["language"] == "urd_Arab" for r in results))

    def test_bm25_language_filter(self):
        results = self.bm25.search(EN_CDG_QUERY, top_k=5, language="eng_Latn")
        self.assertTrue(results)
        self.assertTrue(all(r.metadata["language"] == "eng_Latn" for r in results))

    def test_hybrid_auto_detects_urdu(self):
        results = self.hybrid.search(UR_CDG_QUERY, top_k=5)
        self.assertTrue(results)
        self.assertTrue(all(r.metadata["language"] == "urd_Arab" for r in results))

    def test_explicit_language_hint_wins(self):
        results = self.hybrid.search(EN_CDG_QUERY, top_k=5, language="urd_Arab")
        self.assertTrue(results)
        self.assertTrue(all(r.metadata["language"] == "urd_Arab" for r in results))

    def test_unknown_language_leaves_results_unfiltered(self):
        results = self.hybrid.search(EN_CDG_QUERY, top_k=5, language="hin_Deva")
        self.assertTrue(results)  # unsupported language must not silently drop all


class TestGroupD_NoiseQueries(unittest.TestCase):
    NOISE = [
        "What is the weather in Tokyo today?",
        "Recipe for chocolate chip cookies?",
        "How do I fix a leaking faucet?",
        "List all US presidents from 1900 to 2000.",
    ]

    def setUp(self):
        self.svc, self.dense, self.bm25 = build_stack(CDG_CHUNKS)
        self.hybrid = HybridRetriever(dense=self.dense, bm25=self.bm25, embeddings=self.svc)

    def test_noise_queries_do_not_crash(self):
        for q in self.NOISE:
            results = self.hybrid.search(q, top_k=5)
            self.assertLessEqual(len(results), 5)
            for r in results:
                self.assertIn("text", r.metadata)
                self.assertEqual(r.metadata["chunk_id"], r.chunk_id)

    def test_noise_query_top_k_bounded(self):
        for k in (1, 3, 10):
            for q in self.NOISE:
                self.assertLessEqual(len(self.hybrid.search(q, top_k=k)), k)

    def test_noise_query_returns_sorted_scores(self):
        for q in self.NOISE:
            results = self.hybrid.search(q, top_k=5)
            scores = [r.score for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True))


class TestGroupE_ComponentCorrectness(unittest.TestCase):
    def setUp(self):
        self.svc, self.dense, self.bm25 = build_stack(CDG_CHUNKS)
        self.hybrid = HybridRetriever(dense=self.dense, bm25=self.bm25, embeddings=self.svc)

    def test_top_k_respected_across_all_retrievers(self):
        qvec = self.svc.embed_query(EN_CDG_QUERY)
        for k in (1, 2, 10):
            self.assertLessEqual(len(self.dense.search(qvec, top_k=k)), k)
            self.assertLessEqual(len(self.bm25.search(EN_CDG_QUERY, top_k=k)), k)
            self.assertLessEqual(len(self.hybrid.search(EN_CDG_QUERY, top_k=k)), k)

    def test_scores_descending(self):
        qvec = self.svc.embed_query(EN_CDG_QUERY)
        for results in (
            self.dense.search(qvec, top_k=5),
            self.bm25.search(EN_CDG_QUERY, top_k=5),
            self.hybrid.search(EN_CDG_QUERY, top_k=5),
        ):
            scores = [r.score for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_query_short_circuits(self):
        # The guarantee lives at the production layer (see test_production_retriever);
        # at the raw layers an empty query must never crash or return junk.
        for q in ("", "   "):
            self.assertEqual(self.bm25.search(q, top_k=5), [])
            self.assertEqual(self.hybrid.search(q, top_k=5).__class__, list)
            self.assertLessEqual(len(self.hybrid.search(q, top_k=5)), 5)

    def test_production_retriever_short_circuits_empty_query(self):
        import tempfile

        from app.retrieval import ProductionRetriever

        svc = _svc()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dense = FAISSStore.build(CDG_CHUNKS, svc.embed_documents([c["text"] for c in CDG_CHUNKS]))
            dense.save(tmp)
            BM25Retriever.build(CDG_CHUNKS).save(tmp)
            retriever = ProductionRetriever(
                strategy="recursive", index_dir=tmp, embeddings=svc, model_name="fake-model"
            )
            self.assertEqual(retriever.search("   ", top_k=5), [])
            self.assertEqual(retriever.search("", top_k=5), [])
            self.assertFalse(retriever.is_loaded)

    def test_bm25_empty_after_tokenization(self):
        # Query with nothing tokenizable must not raise or return junk.
        self.assertEqual(self.bm25.search("!!!...", top_k=5), [])

    def test_hybrid_marks_fused_results_as_selected(self):
        results = self.hybrid.search(EN_CDG_QUERY, top_k=5)
        self.assertTrue(results)
        self.assertTrue(all(r.selected_by_hybrid for r in results))

    def test_source_retrievers_do_not_mark_selected(self):
        qvec = self.svc.embed_query(EN_CDG_QUERY)
        for r in self.dense.search(qvec, top_k=5):
            self.assertFalse(r.selected_by_hybrid)
        for r in self.bm25.search(EN_CDG_QUERY, top_k=5):
            self.assertFalse(r.selected_by_hybrid)

    def test_default_chunks_fixture_still_works(self):
        svc, dense, bm25 = build_small_stack()
        hybrid = HybridRetriever(dense=dense, bm25=bm25, embeddings=svc)
        results = hybrid.search("Where is the capital of France?", top_k=5)
        self.assertIn("c1", [r.chunk_id for r in results])


class TestAuditScript(unittest.TestCase):
    """The production audit table logic, exercised on a tiny fake index."""

    def setUp(self):
        import importlib.util

        # scripts/ has no __init__.py; load the module by path so its app.*
        # imports resolve and no real embedding model is constructed.
        spec = importlib.util.spec_from_file_location(
            "evaluate_retrieval_script", BASE_DIR / "scripts" / "evaluate_retrieval.py"
        )
        self.ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.ev)

    def test_audit_ranks_known_evidence(self):
        import tempfile

        svc = _svc()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dense = FAISSStore.build(CDG_CHUNKS, svc.embed_documents([c["text"] for c in CDG_CHUNKS]))
            dense.save(tmp)
            BM25Retriever.build(CDG_CHUNKS).save(tmp)

            report = self.ev.run_audit(tmp, model="fake", batch_size=4, top_k=10, embeddings=svc)

        by_label = {row["label"]: row for row in report["rows"]}
        # Known evidence: the CDG chunk (msmarco_xi_000917, passage 3) exists in
        # the index, so it must be found by every retriever.
        for label in ("A1", "A2", "A3"):
            self.assertTrue(by_label[label]["evidence_present"], label)
            self.assertEqual(by_label[label]["status"], "PASS", label)
            for mode in ("bm25_rank", "dense_rank", "hybrid_rank"):
                self.assertIsNotNone(by_label[label][mode], (label, mode))
        # Missing evidence: no chunk answers these, so the audit must report
        # the coverage gap rather than a fabricated rank.
        for label in ("D1", "D2"):
            self.assertFalse(by_label[label]["evidence_present"], label)
            self.assertEqual(by_label[label]["status"], "NO-EVIDENCE", label)
            self.assertIsNone(by_label[label]["hybrid_rank"], label)


class TestModelReuse(unittest.TestCase):
    """The embedding model must load exactly once and be reused per query."""

    def test_five_queries_load_the_model_once(self):
        svc = _svc()
        self.assertEqual(svc.load_count, 0)
        for _ in range(5):
            svc.embed_query(EN_CDG_QUERY)
        self.assertEqual(svc.load_count, 1)

    def test_documents_and_queries_share_one_model(self):
        svc = _svc()
        svc.embed_documents([c["text"] for c in CDG_CHUNKS])
        svc.embed_query(EN_CDG_QUERY)
        svc.embed_query(UR_CDG_QUERY)
        self.assertEqual(svc.load_count, 1)


class TestSmokeScript(unittest.TestCase):
    """The --smoke battery logic, exercised on a tiny fake index."""

    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "evaluate_retrieval_script_smoke", BASE_DIR / "scripts" / "evaluate_retrieval.py"
        )
        self.ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.ev)

    def test_smoke_exactly_five_queries_one_model_load(self):
        import tempfile

        svc = _svc()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dense = FAISSStore.build(CDG_CHUNKS, svc.embed_documents([c["text"] for c in CDG_CHUNKS]))
            dense.save(tmp)
            BM25Retriever.build(CDG_CHUNKS).save(tmp)

            report = self.ev.run_smoke(tmp, model="fake", batch_size=4, top_k=10, embeddings=svc)

        self.assertEqual(len(report["rows"]), 5)
        self.assertEqual(report["model_loads"], 1)
        by_label = {row["label"]: row for row in report["rows"]}
        # Known evidence: CDG (msmarco_xi_000917, passage 3) is indexed, so it
        # must be found by the hybrid retriever.
        for label in ("S1", "S2", "S3"):
            self.assertTrue(by_label[label]["evidence_present"], label)
            self.assertEqual(by_label[label]["status"], "PASS", label)
            self.assertIsNotNone(by_label[label]["hybrid_rank"], label)
        # Missing evidence: the corpus has nothing answering these, so the
        # smoke must report a coverage gap, not a fabricated failure.
        for label in ("S4", "S5"):
            self.assertFalse(by_label[label]["evidence_present"], label)
            self.assertEqual(by_label[label]["status"], "NOT_APPLICABLE", label)
            self.assertIsNone(by_label[label]["hybrid_rank"], label)


if __name__ == "__main__":
    unittest.main()

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
from app.retrieval.filters import detect_script_language, filter_results
from tests.support import FakeEncoder, make_chunk


HINDI_CHUNKS = [
    make_chunk("h1", "भारत की राजधानी नई दिल्ली है।", language="hin_Deva", record_id="hrec1", passage_index=0),
    make_chunk("h2", "दिल्ली एक प्रमुख शहर और देश की राजधानी है।", language="hin_Deva", record_id="hrec1", passage_index=0),
    make_chunk("h3", "गंगा भारत की सबसे लंबी नदी है।", language="hin_Deva", record_id="hrec2", passage_index=0),
    make_chunk("h4", "हिमालय पर्वत श्रृंखला भारत में स्थित है।", language="hin_Deva", record_id="hrec3", passage_index=0),
    make_chunk("h5", "यह एक हिंदी दस्तावेज़ है।", language="hin_Deva", record_id="hrec4", passage_index=0),
]


class TestHindiRetrieval(unittest.TestCase):
    def setUp(self):
        self.svc = EmbeddingService(encoder=FakeEncoder(32), use_query_prefix=False, use_doc_prefix=False)
        self.matrix = embed_in_batches([c["text"] for c in HINDI_CHUNKS], self.svc.embed_documents, 4)
        self.dense = FAISSStore.build(HINDI_CHUNKS, self.matrix, model_name="fake-hindi-model")
        self.bm25 = BM25Retriever.build(HINDI_CHUNKS)
        self.hybrid = HybridRetriever(self.dense, self.bm25, self.svc)

    def test_detect_hindi_script(self):
        self.assertEqual(detect_script_language("भारत की राजधानी क्या है?"), "hin_Deva")

    def test_hindi_faiss_index_loads_and_filters(self):
        q = self.svc.embed_query("भारत की राजधानी क्या है?")
        results = self.dense.search(q, top_k=3, language="hin_Deva")
        self.assertTrue(results)
        self.assertTrue(all(r.metadata["language"] == "hin_Deva" for r in results))

    def test_hindi_bm25_returns_only_hindi(self):
        results = self.bm25.search("भारत की राजधानी", top_k=3, language="hin_Deva")
        self.assertTrue(results)
        self.assertTrue(all(r.metadata["language"] == "hin_Deva" for r in results))

    def test_hybrid_hindi_query_returns_hindi_only_results(self):
        results = self.hybrid.search("भारत की राजधानी", top_k=3, language="hin_Deva")
        self.assertTrue(results)
        self.assertTrue(all(r.metadata["language"] == "hin_Deva" for r in results))
        self.assertLessEqual(len(results), 3)

    def test_hindi_filter_rejects_english_and_urdu(self):
        mixed = [
            make_chunk("x1", "Paris is in France.", language="eng_Latn", record_id="x1"),
            make_chunk("x2", "یہ اردو ہے۔", language="urd_Arab", record_id="x2"),
            make_chunk("x3", "भारत की राजधानी नई दिल्ली है।", language="hin_Deva", record_id="x3"),
        ]
        results = filter_results(
            [
                type("R", (), {"metadata": _})() for _ in mixed
            ],
            "hin_Deva",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["language"], "hin_Deva")

    def test_hindi_empty_query_short_circuits(self):
        self.assertEqual(self.bm25.search("   ", top_k=5, language="hin_Deva"), [])
        self.assertEqual(self.hybrid.search("", top_k=5, language="hin_Deva"), [])

    def test_hindi_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            dense_dir = Path(tmp) / "hindi"
            self.dense.save(dense_dir)
            self.bm25.save(dense_dir)
            loaded_dense = FAISSStore.load(dense_dir)
            loaded_bm25 = BM25Retriever.load(dense_dir)
            self.assertEqual(loaded_dense.index.ntotal, len(HINDI_CHUNKS))
            self.assertEqual(len(loaded_bm25.chunks), len(HINDI_CHUNKS))


if __name__ == "__main__":
    unittest.main()

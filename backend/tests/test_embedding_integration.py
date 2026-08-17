"""Integration tests for the real embedding model.

These load the actual ``intfloat/multilingual-e5-small`` SentenceTransformer
model and are excluded from the default unit-test run so the suite never hangs
(or downloads) because of a real model. Run them explicitly with::

    VOICE_RAG_INTEGRATION=1 ./venv/bin/python -m unittest tests.test_embedding_integration

The model is loaded once per process and reused, mirroring production.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from app.retrieval import EmbeddingService

MODEL = "intfloat/multilingual-e5-small"


@unittest.skipUnless(os.environ.get("VOICE_RAG_INTEGRATION"), "set VOICE_RAG_INTEGRATION=1 to run")
class TestRealEmbeddingService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = EmbeddingService(model_name=MODEL, batch_size=8)

    def test_model_loads_and_reused(self):
        v = self.svc.embed_documents(["hello world"])
        self.assertEqual(v.shape, (1, 384))
        self.assertEqual(self.svc.load_count, 1)
        v2 = self.svc.embed_documents(["another document"])
        self.assertEqual(self.svc.load_count, 1)  # loaded exactly once

    def test_query_embedding_consistent_dimensions(self):
        q = self.svc.embed_query("what is a corporation?")
        d = self.svc.embed_documents(["a corporation is a company"])
        self.assertEqual(q.shape, (384,))
        self.assertEqual(d.shape, (1, 384))
        self.assertEqual(self.svc.dimension, 384)

    def test_normalized(self):
        v = self.svc.embed_documents(["hello"])
        self.assertAlmostEqual(float(np.linalg.norm(v[0])), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

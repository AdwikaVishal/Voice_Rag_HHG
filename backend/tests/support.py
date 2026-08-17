"""Shared fixtures for the retrieval unit tests.

Loaded as a plain top-level module by ``unittest discover -s tests``; import
with ``from support import ...``. No real embedding model is ever loaded here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.retrieval import EmbeddingService, FAISSStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import embed_in_batches

BASE_DIR = Path(__file__).resolve().parent.parent


def make_chunk(chunk_id, text, language="eng_Latn", record_id="rec", passage_index=0):
    return {
        "chunk_id": chunk_id,
        "record_id": record_id,
        "query_id": 1,
        "text": text,
        "language": language,
        "source_lang": "eng_Latn",
        "target_lang": "urd_Arab",
        "chunk_position": 0,
        "total_chunks": 1,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "passage_index": passage_index,
        "is_selected": 0,
        "text_field": "english_text",
        "char_count": len(text),
        "token_count": len(text.split()),
    }


CHUNKS = [
    make_chunk("c1", "The capital of France is Paris. France is in Europe."),
    make_chunk("c2", "The Eiffel Tower is a famous landmark in Paris."),
    make_chunk("c3", "Bears hibernate during the winter months."),
    make_chunk("c4", "India has many official languages."),
    make_chunk(
        "c5",
        "کارپوریشن ایک کمپنی ہے جو قانون میں ایک ادارے کے طور پر کام کرتی ہے۔",
        language="urd_Arab",
        record_id="rec2",
    ),
]

QUERIES = ["Where is the capital of France?", "How do corporations work?"]


class FakeEncoder:
    """Deterministic fake encoder: hashes text into a fixed-dim vector."""

    def __init__(self, dim=32):
        self.dim = dim
        self.encode_calls = 0

    def encode(self, texts, **kwargs):
        self.encode_calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            h = abs(hash(text))
            rng = np.random.default_rng(h % (2**32))
            out[i] = rng.standard_normal(self.dim)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def get_embedding_dimension(self):
        return self.dim


def make_service(dim=32):
    return EmbeddingService(encoder=FakeEncoder(dim), use_query_prefix=True, use_doc_prefix=True)


def build_small_stack(dim=32, chunks=None):
    chunks = chunks or CHUNKS
    svc = make_service(dim)
    matrix = embed_in_batches([c["text"] for c in chunks], svc.embed_documents, 4)
    dense = FAISSStore.build(chunks, matrix)
    bm25 = BM25Retriever.build(chunks)
    return svc, dense, bm25

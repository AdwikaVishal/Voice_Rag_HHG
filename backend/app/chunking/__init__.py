"""Chunking strategies for the Voice RAG pipeline.

Four interchangeable strategies are provided:

* ``fixed``     — fixed-size token chunks with overlap (baseline/control)
* ``sentence``  — sentence-boundary-aware greedy combination
* ``recursive`` — recursive paragraph -> sentence -> token splitting
* ``metadata``  — sentence-aware splitting enriched with language metadata

Use :func:`create_chunker` to build a chunker by name.
"""

from .base import BaseChunker
from .factory import available_strategies, create_chunker
from .fixed import FixedSizeChunker
from .metadata import MetadataAwareChunker
from .models import Chunk
from .recursive import RecursiveChunker
from .sentence import SentenceChunker

__all__ = [
    "BaseChunker",
    "Chunk",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "MetadataAwareChunker",
    "create_chunker",
    "available_strategies",
]

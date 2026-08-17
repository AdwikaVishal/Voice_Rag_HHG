"""Voice/text query pipeline (Segments 4B + 4C + 4D + 4 + 5 + 5.1): input ->
guardrail -> retrieval -> answerability gate -> answer-support verification ->
router -> LLM/general -> TTS.

The pipeline reuses the existing lazy STT, production-retriever, LLM and TTS
singletons, so the Whisper model, embedding model, FAISS index, BM25 index and
the LLM backend are each loaded once per process. The Segment 4 answerability
gate decides whether the retrieved context can answer the query at all; the
Segment 5 router then picks the answer route: a grounded RAG answer, a
cautious clarifying RAG answer, a general-knowledge fallback (query only), or
the pure abstention. The Segment 5.1 verifier re-checks the supporting
evidence after an ANSWERABLE verdict — retrieval relevance is not answer
support — and downgrades to general knowledge when the evidence does not
directly answer the exact question.
"""

from __future__ import annotations

from functools import lru_cache

from .models import GenerationInfo, GuardrailInfo, SourceInfo, Timings, Usage, VoiceQueryResponse
from .query_pipeline import PipelineStageError, QueryPipeline

__all__ = [
    "QueryPipeline",
    "PipelineStageError",
    "VoiceQueryResponse",
    "GenerationInfo",
    "GuardrailInfo",
    "SourceInfo",
    "Timings",
    "Usage",
    "get_query_pipeline",
]


@lru_cache(maxsize=1)
def get_query_pipeline() -> QueryPipeline:
    """Process-wide singleton pipeline used by the FastAPI layer."""
    # Defer importing the TTS singleton until the function is called so that
    # importing the pipeline package does not require heavy multimedia
    # dependencies (PyAV, edge_tts, faster-whisper) during test collection.
    from ..tts import get_tts_service

    return QueryPipeline(tts=get_tts_service())

"""RAG answerability gate (Segment 4) + fallback router (Segment 5) +
answer-support verification (Segment 5.1).

Segment 4 — deterministic, model-free layer between retrieval and the LLM that
decides whether the retrieved context can answer a query at all:

* :class:`AnswerabilityEvaluator` — the gate itself (pure function of query,
  language and retrieved results)
* :class:`AnswerabilityDecision`  — the three-way verdict
  (ANSWERABLE / UNCERTAIN / UNANSWERABLE_FROM_RAG) with confidence, reason and
  supporting chunks
* :func:`api_status` — maps the internal verdict to the stable API status
  vocabulary (``grounded`` / ``uncertain`` / ``insufficient_evidence``)

Segment 5 — controlled general-knowledge fallback for queries the corpus
cannot answer:

* :class:`FallbackRouter`  — routes the gate verdict to
  RAG_GROUNDED / RAG_UNCERTAIN / GENERAL_KNOWLEDGE / ABSTAIN
* :class:`GeneralKnowledgeProvider` — answers the USER QUERY from general
  knowledge (never receives retrieved chunks)

Segment 5.1 — answer-support verification:

* :class:`AnswerSupportVerifier` — a deterministic, model-free re-check of the
  supporting evidence after the gate says ANSWERABLE: retrieval relevance is NOT
  answer support, so evidence that does not DIRECTLY answer the exact question
  (e.g. a historical passage about Versailles for "What is the capital of
  France?") is rejected and routed to general knowledge instead.

No embedding model, index or provider is ever loaded here.
"""

from __future__ import annotations

from functools import lru_cache

from .answerability import AnswerabilityEvaluator
from .general import GeneralKnowledgeProvider, GeneralKnowledgeResponse
from .models import AnswerabilityDecision, AnswerabilityStatus, api_status
from .router import (
    FallbackRouter,
    Route,
    get_fallback_router,
    route,
    source_for_route,
)
from .verifier import AnswerSupportVerifier, SupportVerdict

__all__ = [
    "AnswerabilityEvaluator",
    "AnswerabilityDecision",
    "AnswerabilityStatus",
    "api_status",
    "Route",
    "FallbackRouter",
    "route",
    "source_for_route",
    "get_fallback_router",
    "GeneralKnowledgeProvider",
    "GeneralKnowledgeResponse",
    "AnswerSupportVerifier",
    "SupportVerdict",
    "get_answerability_evaluator",
    "get_general_knowledge_provider",
    "get_answer_support_verifier",
]


@lru_cache(maxsize=1)
def get_answerability_evaluator() -> AnswerabilityEvaluator:
    """Process-wide singleton evaluator (stateless; reads config defaults)."""
    return AnswerabilityEvaluator()


@lru_cache(maxsize=1)
def get_general_knowledge_provider() -> GeneralKnowledgeProvider:
    """Process-wide singleton general-knowledge provider.

    Resolves the shared LLM service lazily on first use, so no model or
    provider backend is loaded at import time.
    """
    return GeneralKnowledgeProvider()


@lru_cache(maxsize=1)
def get_answer_support_verifier() -> AnswerSupportVerifier:
    """Process-wide singleton answer-support verifier.

    The verifier is deterministic (pure token matching over the supporting
    evidence), so no model or provider backend is loaded at import time.
    """
    return AnswerSupportVerifier()

"""Typed models for the RAG answerability gate (Segment 4).

The gate is the deterministic layer between retrieval and the LLM. It turns a
list of retrieved chunks into one of three decisions:

* ``ANSWERABLE``            — the context plausibly contains the answer
* ``UNCERTAIN``             — some evidence, not enough confidence for a firm answer
* ``UNANSWERABLE_FROM_RAG`` — the context must NOT be used to answer

The internal decision status maps to the stable API status vocabulary the
frontend and downstream consumers rely on:

    ANSWERABLE            -> ``grounded``
    UNCERTAIN             -> ``uncertain``
    UNANSWERABLE_FROM_RAG -> ``insufficient_evidence``
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerabilityStatus:
    """Machine-readable gate decisions (kept separate from the API status)."""

    ANSWERABLE = "ANSWERABLE"
    UNCERTAIN = "UNCERTAIN"
    UNANSWERABLE_FROM_RAG = "UNANSWERABLE_FROM_RAG"


# Internal decision -> stable API status vocabulary.
_API_STATUS = {
    AnswerabilityStatus.ANSWERABLE: "grounded",
    AnswerabilityStatus.UNCERTAIN: "uncertain",
    AnswerabilityStatus.UNANSWERABLE_FROM_RAG: "insufficient_evidence",
}


def api_status(status: str) -> str:
    """Map an internal decision to the API ``status`` string.

    Unknown values fall back to ``insufficient_evidence`` so the API never
    exposes an untyped status.
    """
    return _API_STATUS.get(status, _API_STATUS[AnswerabilityStatus.UNANSWERABLE_FROM_RAG])


class AnswerabilityDecision(BaseModel):
    """Verdict of one answerability evaluation.

    ``confidence`` is a normalized [0, 1] score combining the deterministic
    signals; it is attenuated toward zero when the answer's content words are
    absent from the context, so a low value reliably accompanies an
    ``UNANSWERABLE_FROM_RAG`` verdict. ``supporting_chunk_ids`` are the chunks
    that actually mention query content terms and are only populated when the
    query is answerable or uncertain — an abstention never cites sources.
    """

    status: str = Field(
        ..., description="One of AnswerabilityStatus: ANSWERABLE | UNCERTAIN | UNANSWERABLE_FROM_RAG."
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = Field("", description="Deterministic reason for the verdict.")
    evidence_count: int = Field(
        0, ge=0, description="Chunks that mention at least one query content term."
    )
    best_score: float = Field(
        0.0, description="Best retrieval score observed (informational; not cross-compared)."
    )
    supporting_chunk_ids: list[str] = Field(
        default_factory=list, description="Chunk ids the answer would be grounded on (empty on abstention)."
    )

"""Typed models for the voice query pipeline (Segments 4B + 4C + 4D)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..llm.models import Usage
from ..schemas import SearchResponse
from ..tts.models import TTSResponse


class GuardrailInfo(BaseModel):
    """Guardrail verdict exposed on the API (no internal implementation detail)."""

    allowed: bool
    reason: Optional[str] = None


class SourceInfo(BaseModel):
    """One retrieved source, with only the metadata worth showing the user.

    Deliberately a projection of the internal :class:`RetrievedChunk` — no
    ``metadata`` dict, no query/passage internals, no raw index details.
    """

    id: str = Field(..., description="Stable chunk/source identifier.")
    score: float = Field(..., description="Hybrid retrieval relevance score.")
    language: Optional[str] = Field(
        None, description="Chunk language in script form (eng_Latn | urd_Arab), or null."
    )
    excerpt: str = Field("", description="Short text excerpt of the source.")


class GenerationInfo(BaseModel):
    """LLM answer-generation block (Segment 4C), including the Segment 4
    answerability verdict from the pre-LLM gate and the Segment 5 route/source.

    ``status`` follows the stable API vocabulary: ``grounded`` (RAG_GROUNDED),
    ``uncertain`` (RAG_UNCERTAIN), ``answered`` (GENERAL_KNOWLEDGE) or
    ``insufficient_evidence`` (ABSTAIN). ``source`` says which mode produced
    the answer: ``rag`` | ``clarification`` | ``general_knowledge`` |
    ``abstained``.
    """

    answer: str
    model: str
    language: Optional[str] = Field(
        None, description="Answer language in script form (eng_Latn | urd_Arab), or null."
    )
    grounded: Optional[bool] = Field(
        None, description="True / False / None (unknown) from the grounding check."
    )
    context_count: int = Field(0, ge=0)
    latency_ms: float = Field(0.0, ge=0)
    abstained: bool = False
    sources: list[SourceInfo] = Field(
        default_factory=list,
        description="Retrieved sources used to answer (source transparency).",
    )
    usage: Optional[Usage] = None
    status: str = Field(
        "grounded",
        description=(
            "Answerability verdict from the Segment 4 gate plus the Segment 5 "
            "general-knowledge route: grounded | uncertain | answered | "
            "insufficient_evidence."
        ),
    )
    source: str = Field(
        "rag",
        description=(
            "Which mode produced the answer: rag | clarification | "
            "general_knowledge | abstained."
        ),
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Gate confidence in [0, 1], or null for general knowledge."
    )
    reason: Optional[str] = Field(
        None, description="Deterministic reason behind the gate verdict."
    )
    evidence_count: int = Field(
        0, ge=0, description="Retrieved chunks mentioning at least one query content term."
    )
    best_score: float = Field(
        0.0, description="Best retrieval score observed (informational)."
    )
    supporting_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk ids the answer is grounded on (empty when abstaining or falling back).",
    )


class Timings(BaseModel):
    """Per-stage latency (ms) for one voice query."""

    stt_ms: float = Field(..., ge=0)
    guardrail_ms: float = Field(..., ge=0)
    retrieval_ms: float = Field(..., ge=0)
    llm_ms: float = Field(..., ge=0)
    grounding_ms: float = Field(0.0, ge=0)
    tts_ms: float = Field(0.0, ge=0)
    total_ms: float = Field(..., ge=0)


class VoiceQueryResponse(BaseModel):
    """Response of ``POST /voice/query``: transcript + guardrail + context +
    answer + TTS metadata.

    ``retrieval`` and ``generation`` are ``None`` when the guardrail rejected
    the input (retrieval and generation never run after a rejection). ``tts``
    is ``None`` when there is no answer to speak (guardrail rejection, empty
    answer, or no TTS backend configured). The retrieval block reuses the
    existing :class:`SearchResponse` schema so the chunk format is identical
    to ``POST /search``.
    """

    transcript: str
    language: Optional[str] = Field(
        None,
        description="Resolved language in script form (eng_Latn | urd_Arab), or null.",
    )
    guardrail: GuardrailInfo
    retrieval: Optional[SearchResponse] = None
    generation: Optional[GenerationInfo] = None
    tts: Optional[TTSResponse] = Field(
        None, description="TTS metadata (never a filesystem path; audio is streamed)."
    )
    timings: Timings

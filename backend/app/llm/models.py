"""Typed models for the LLM answer-generation layer (Segment 4C)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Token usage reported by the provider (only if the provider supplies it)."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMResponse(BaseModel):
    """Result of one grounded generation call.

    ``grounded`` is decided by the deterministic grounding check, never
    assumed true just because retrieval returned results:

    * ``True``    — answer has meaningful lexical overlap with the context
    * ``False``   — no context, empty answer, abstention, or no overlap
    * ``None``    — cannot be validated confidently (unknown)

    ``abstained`` is True when the answer is an explicit uncertainty response
    (either because retrieval was empty or the LLM said the context was
    insufficient).
    """

    answer: str
    model: str
    grounded: Optional[bool] = Field(None, description="True / False / None (unknown).")
    context_count: int = Field(0, ge=0)
    language: Optional[str] = Field(None, description="Script form (eng_Latn | urd_Arab), or null.")
    latency_ms: float = Field(0.0, ge=0)
    abstained: bool = False
    usage: Optional[Usage] = None

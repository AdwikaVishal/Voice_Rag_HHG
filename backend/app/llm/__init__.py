"""LLM answer generation (Segment 4C): grounded RAG answers.

Exposes :class:`LLMService` — the replaceable, environment-configured provider
abstraction with deterministic grounding validation — and
:data:`get_llm_service`, the process-wide singleton used by the pipeline.
"""

from __future__ import annotations

from functools import lru_cache

from .models import LLMResponse, Usage
from .prompts import ABSTENTION_EN, ABSTENTION_UR, build_messages, format_context
from .service import (
    LLMProvider,
    LLMProviderError,
    LLMService,
    OllamaProvider,
    OpenAIProvider,
    is_abstention,
    make_provider,
    validate_grounding,
)

__all__ = [
    "LLMService",
    "LLMResponse",
    "Usage",
    "LLMProvider",
    "LLMProviderError",
    "OllamaProvider",
    "OpenAIProvider",
    "make_provider",
    "validate_grounding",
    "is_abstention",
    "build_messages",
    "format_context",
    "ABSTENTION_EN",
    "ABSTENTION_UR",
    "get_llm_service",
]


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """Process-wide singleton LLM service (stateless HTTP backend)."""
    return LLMService()

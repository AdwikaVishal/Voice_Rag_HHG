"""General-knowledge fallback provider (Segment 5).

Used when the Segment 4 answerability gate finds no usable RAG evidence. The
provider answers the USER QUERY from the model's own knowledge — it never
receives retrieved chunks, so irrelevant RAG context can never leak into the
fallback answer.

The provider is deliberately provider-agnostic and reuses the project's
existing LLM configuration/backend (``LLM_PROVIDER`` / ``LLM_MODEL`` ...) via
:class:`LLMService`: no separate model, client or configuration is created for
the fallback. Tests inject a fake provider/service; nothing here ever requires
network access at import time.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from pydantic import BaseModel, Field

from ..llm import LLMService, get_llm_service
from ..llm.models import Usage
from ..llm.prompts import build_general_messages
from ..llm.service import LLMProviderError, is_abstention

logger = logging.getLogger("rag.general")


class GeneralKnowledgeResponse(BaseModel):
    """Answer produced by the general-knowledge provider (Segment 5).

    ``grounded`` is always ``False`` by design: this answer is NOT grounded in
    retrieved evidence. ``abstained`` is ``True`` only when the model itself
    declined to answer. ``usage`` and ``latency_ms`` come from the backend.
    """

    answer: str
    model: str
    language: Optional[str] = Field(None, description="Script form (eng_Latn | urd_Arab), or null.")
    latency_ms: float = Field(0.0, ge=0)
    abstained: bool = False
    usage: Optional[Usage] = None


class GeneralKnowledgeProvider:
    """Answers a query from general knowledge (no retrieved context).

    ``answer(query, language)`` builds a dedicated general-knowledge prompt
    containing ONLY the user query and calls the configured chat backend.
    Provider failures surface as :class:`LLMProviderError` (mapped to 502 by
    the API layer), exactly like the RAG LLM path.
    """

    def __init__(
        self,
        llm: Optional[LLMService] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        # ``llm`` is resolved lazily so constructing the provider never loads a
        # model or touches the network.
        self._llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def _backend(self) -> LLMService:
        """The underlying chat backend (the injected LLM service)."""
        if self._llm is None:
            self._llm = get_llm_service()
        return self._llm

    @property
    def provider_name(self) -> str:
        """Name of the underlying provider backend (e.g. ``ollama``)."""
        backend = self._backend
        return str(
            getattr(getattr(backend, "provider", None), "name", None)
            or getattr(backend, "provider_name", None)
            or "unknown"
        )

    def answer(
        self,
        query: str,
        language: Optional[str] = None,
    ) -> GeneralKnowledgeResponse:
        """Answer ``query`` from general knowledge (no RAG context).

        ``language`` is the script-form code of the query (``eng_Latn`` /
        ``urd_Arab``) and steers the answer language; ``None`` falls back to
        answering in the query's language.
        """
        query = (query or "").strip()
        backend = self._backend
        model = self.model or getattr(backend, "model", None) or ""
        temperature = self.temperature if self.temperature is not None else getattr(backend, "temperature", 0.2)
        max_tokens = self.max_tokens if self.max_tokens is not None else getattr(backend, "max_tokens", 300)
        messages = build_general_messages(query, language)

        start = time.perf_counter()
        try:
            result = backend.provider.chat(
                model,
                messages,
                temperature,
                max_tokens,
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            logger.error("General-knowledge provider failed for model=%s: %s", model, exc)
            raise LLMProviderError(
                "General-knowledge provider request failed."
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        answer = (result.content or "").strip()
        return GeneralKnowledgeResponse(
            answer=answer,
            model=result.model or model,
            language=language,
            latency_ms=round(latency_ms, 2),
            abstained=is_abstention(answer),
            usage=result.usage,
        )

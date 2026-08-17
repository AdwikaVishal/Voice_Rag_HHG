"""LLM provider abstraction, grounded generation, and deterministic grounding
validation (Segment 4C).

The provider is replaceable and selected purely from environment variables
(``LLM_PROVIDER`` / ``LLM_MODEL`` / ``LLM_API_KEY``). The default provider is
Ollama (local, no API key). An OpenAI-compatible backend is available for
environments that expose ``LLM_API_KEY``.

Security: the API key is read from the environment at construction time, is
never logged, never returned by the API, and never written to source. Only the
provider name and model are logged.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin

from ..config import (
    LLM_BASE_URL,
    LLM_MAX_CONTEXT_CHARS,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_S,
)
from .models import LLMResponse, Usage
from .prompts import ABSTENTION_EN, ABSTENTION_UR, build_messages

logger = logging.getLogger("llm")


class LLMProviderError(RuntimeError):
    """Raised when the LLM provider fails; never exposes secrets."""


@dataclass
class ProviderResult:
    content: str
    model: str
    usage: Optional[Usage] = None


class LLMProvider:
    """Transport-agnostic base class for a chat-completions provider."""

    name: str = "provider"

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Local Ollama backend (``POST /api/chat``). No API key needed."""

    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def chat(self, model, messages, temperature, max_tokens) -> ProviderResult:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = str(data.get("message", {}).get("content", "") or "")
        usage = Usage(
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )
        return ProviderResult(content=content, model=str(data.get("model", model)), usage=usage)


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible chat-completions backend (key from environment only)."""

    name = "openai"

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = float(timeout_s)

    def chat(self, model, messages, temperature, max_tokens) -> ProviderResult:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        request = urllib.request.Request(
            urljoin(f"{self.base_url}/", "chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = str(data["choices"][0]["message"]["content"]) if data.get("choices") else ""
        usage_data = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens"),
            completion_tokens=usage_data.get("completion_tokens"),
            total_tokens=usage_data.get("total_tokens"),
        )
        return ProviderResult(content=content, model=str(data.get("model", model)), usage=usage)


def make_provider(
    name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> LLMProvider:
    """Build a provider from configuration (or explicit overrides)."""
    provider_name = (name or LLM_PROVIDER).strip().lower()
    if provider_name == "ollama":
        return OllamaProvider(
            base_url=base_url or LLM_BASE_URL or "http://localhost:11434",
            timeout_s=timeout_s if timeout_s is not None else LLM_TIMEOUT_S,
        )
    if provider_name in ("openai", "openai_compatible"):
        return OpenAIProvider(
            base_url=base_url or LLM_BASE_URL or "https://api.openai.com/v1",
            api_key=api_key if api_key is not None else os.environ.get("LLM_API_KEY", ""),
            timeout_s=timeout_s if timeout_s is not None else LLM_TIMEOUT_S,
        )
    raise ValueError(f"Unsupported LLM provider: {provider_name!r}")


# --------------------------------------------------------------------------
# Deterministic grounding validation
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)

_EN_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of",
        "in", "on", "at", "by", "for", "and", "or", "but", "it", "its", "this",
        "that", "what", "which", "who", "how", "does", "do", "did", "not",
        "no", "yes", "with", "from", "as", "about", "you", "your", "my", "me",
    }
)

_UR_STOPWORDS = frozenset(
    {
        "کا", "کی", "کے", "کو", "سے", "میں", "پر", "اور", "ہے", "ہیں",
        "تھا", "تھی", "تھے", "ایک", "اس", "یہ", "وہ", "بھی", "نہیں",
        "کیا", "کون", "کہاں", "کیوں", "جو", "نے", "میں", "لیے", "کہ",
        "تم", "آپ", "میرے", "میری",
    }
)

_ERROR_MARKERS = (
    "internal server error", "api error", "model not found", "rate limit exceeded",
    "unable to generate", "could not produce an answer", "invalid api key",
    "request failed", "timeout",
)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").casefold()))


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def is_abstention(answer: str) -> bool:
    """Whether the answer is the explicit uncertainty message (EN or UR)."""
    normalized = _normalize_ws(answer).rstrip(".")
    for message in (ABSTENTION_EN, ABSTENTION_UR):
        candidate = _normalize_ws(message).rstrip(".")
        if normalized == candidate or candidate in normalized:
            return True
    return False


def _is_error_answer(answer: str) -> bool:
    lowered = (answer or "").casefold()
    return any(marker in lowered for marker in _ERROR_MARKERS)


def _overlap_ratio(answer: str, context_text: str) -> float:
    answer_tokens = _tokens(answer) - _EN_STOPWORDS - _UR_STOPWORDS
    if not answer_tokens:
        return 0.0
    context_tokens = _tokens(context_text)
    matched = answer_tokens & context_tokens
    return len(matched) / len(answer_tokens)


def validate_grounding(
    answer: str,
    context: list[str],
    language: Optional[str] = None,
    overlap_threshold: float = 0.4,
) -> Optional[bool]:
    """Deterministic lexical grounding check.

    Returns:
        ``True``     — answer has meaningful overlap with the context
        ``False``    — no usable context, empty/error/abstention answer, or no overlap
        ``None``     — partial overlap only (cannot validate confidently)
    """
    usable = [(c or "").strip() for c in context if (c or "").strip()]
    if not usable:
        return False
    if not (answer or "").strip():
        return False
    if is_abstention(answer):
        return False
    if _is_error_answer(answer):
        return False

    context_text = "\n".join(usable)
    ratio = _overlap_ratio(answer, context_text)
    if ratio >= overlap_threshold:
        return True
    if ratio > 0.0:
        return None
    return False


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------


class LLMService:
    """Grounded answer generation: prompt -> provider -> grounding check.

    Components can be injected for tests; in production ``provider`` is built
    from environment configuration and every call is stateless (no model is
    loaded in this process — the model lives in the provider backend).
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_context_chars: Optional[int] = None,
    ) -> None:
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
        self.max_context_chars = (
            max_context_chars if max_context_chars is not None else LLM_MAX_CONTEXT_CHARS
        )
        self.timeout_s = timeout_s if timeout_s is not None else LLM_TIMEOUT_S
        self.provider = provider or make_provider(
            base_url=base_url, api_key=api_key, timeout_s=self.timeout_s
        )
        self.provider_name = self.provider.name

    def _abstain(self, language: Optional[str]) -> LLMResponse:
        message = ABSTENTION_UR if language == "urd_Arab" else ABSTENTION_EN
        return LLMResponse(
            answer=message,
            model=self.model,
            grounded=False,
            context_count=0,
            language=language,
            latency_ms=0.0,
            abstained=True,
            usage=None,
        )

    def generate(
        self,
        query: str,
        context: list[str],
        language: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a grounded answer for ``query`` from ``context``.

        ``context`` is a list of retrieved chunk texts. If there is no usable
        context the provider is never called and an abstention is returned.
        """
        context_texts = [(c or "").strip() for c in context]
        usable = [c for c in context_texts if c]
        if not usable:
            logger.info("No usable context; abstaining without calling the LLM.")
            return self._abstain(language)

        messages = build_messages(query, context_texts, language, self.max_context_chars)
        start = time.perf_counter()
        try:
            result = self.provider.chat(
                self.model, messages, self.temperature, self.max_tokens
            )
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            logger.error("LLM provider %s failed for model=%s: %s", self.provider_name, self.model, exc)
            raise LLMProviderError(
                f"LLM provider '{self.provider_name}' request failed."
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        answer = (result.content or "").strip()
        grounded = validate_grounding(answer, usable, language)
        return LLMResponse(
            answer=answer,
            model=result.model or self.model,
            grounded=grounded,
            context_count=len(usable),
            language=language,
            latency_ms=round(latency_ms, 2),
            abstained=is_abstention(answer),
            usage=result.usage,
        )

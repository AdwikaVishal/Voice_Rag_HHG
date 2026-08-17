"""Segment 5 — tests for the fallback router and the general-knowledge provider.

Everything is deterministic: the router is a pure function of the gate verdict,
and the general provider is tested against a fake chat backend. No real
embedding model, index or LLM provider is ever loaded.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.llm import LLMProviderError  # noqa: E402
from app.llm.models import Usage  # noqa: E402
from app.llm.service import ProviderResult  # noqa: E402
from app.rag import (  # noqa: E402
    AnswerabilityDecision,
    AnswerabilityStatus,
    FallbackRouter,
    GeneralKnowledgeProvider,
    Route,
    route,
    source_for_route,
)
from app.rag.general import GeneralKnowledgeResponse  # noqa: E402

CAPITAL_FRANCE = "What is the capital of France?"
URDU_QUERY = "فرانس کا دارالحکومت کیا ہے؟"


def decision(status, confidence=0.5, reason="test", evidence_count=0, supporting=()):
    return AnswerabilityDecision(
        status=status,
        confidence=confidence,
        reason=reason,
        evidence_count=evidence_count,
        best_score=1.0,
        supporting_chunk_ids=list(supporting),
    )


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------


class TestRouter(unittest.TestCase):
    def test_answerable_routes_to_rag_grounded(self):
        self.assertEqual(
            route(decision(AnswerabilityStatus.ANSWERABLE, confidence=0.8)),
            Route.RAG_GROUNDED,
        )
        self.assertEqual(source_for_route(Route.RAG_GROUNDED), "rag")

    def test_uncertain_routes_to_rag_uncertain_clarification(self):
        self.assertEqual(
            route(decision(AnswerabilityStatus.UNCERTAIN, confidence=0.55)),
            Route.RAG_UNCERTAIN,
        )
        self.assertEqual(source_for_route(Route.RAG_UNCERTAIN), "clarification")

    def test_unanswerable_routes_to_general_knowledge(self):
        self.assertEqual(
            route(decision(AnswerabilityStatus.UNANSWERABLE_FROM_RAG, confidence=0.2)),
            Route.GENERAL_KNOWLEDGE,
        )
        self.assertEqual(source_for_route(Route.GENERAL_KNOWLEDGE), "general_knowledge")

    def test_unanswerable_routes_to_abstain_when_fallback_disabled(self):
        self.assertEqual(
            route(
                decision(AnswerabilityStatus.UNANSWERABLE_FROM_RAG, confidence=0.2),
                enable_general_knowledge=False,
            ),
            Route.ABSTAIN,
        )
        self.assertEqual(source_for_route(Route.ABSTAIN), "abstained")

    def test_routing_never_depends_on_retrieval_having_results(self):
        # A verdict is the ONLY routing input; empty retrieval that the gate
        # rejects still routes to general knowledge (not to abstain), and a
        # strong verdict routes to RAG regardless of score magnitude.
        empty = decision(AnswerabilityStatus.UNANSWERABLE_FROM_RAG, confidence=0.0)
        self.assertEqual(route(empty), Route.GENERAL_KNOWLEDGE)
        self.assertEqual(
            route(empty, enable_general_knowledge=False), Route.ABSTAIN
        )

    def test_fallback_router_honors_config_flag(self):
        enabled = FallbackRouter(enable_general_knowledge=True)
        disabled = FallbackRouter(enable_general_knowledge=False)
        d = decision(AnswerabilityStatus.UNANSWERABLE_FROM_RAG)
        self.assertEqual(enabled.route(d), Route.GENERAL_KNOWLEDGE)
        self.assertEqual(disabled.route(d), Route.ABSTAIN)
        self.assertEqual(enabled.route(decision(AnswerabilityStatus.ANSWERABLE)), Route.RAG_GROUNDED)
        self.assertEqual(disabled.route(decision(AnswerabilityStatus.ANSWERABLE)), Route.RAG_GROUNDED)

    def test_unknown_route_falls_back_to_rag_source(self):
        self.assertEqual(source_for_route("NOT_A_ROUTE"), "rag")


# --------------------------------------------------------------------------
# General-knowledge provider
# --------------------------------------------------------------------------


class FakeChatProvider:
    name = "fake"

    def __init__(self, content="The capital of France is Paris.", model="fake-model", usage=None):
        self.content = content
        self.model = model
        self.usage = usage or Usage(prompt_tokens=5, completion_tokens=3)
        self.calls: list[tuple] = []

    def chat(self, model, messages, temperature, max_tokens):
        self.calls.append((model, list(messages), temperature, max_tokens))
        return ProviderResult(content=self.content, model=self.model, usage=self.usage)


class FakeBackend:
    def __init__(self, provider, model="fake-model", temperature=0.2, max_tokens=300):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens


class TestGeneralKnowledgeProvider(unittest.TestCase):
    def test_answer_calls_backend_and_returns_response(self):
        backend = FakeBackend(FakeChatProvider())
        provider = GeneralKnowledgeProvider(llm=backend)
        resp = provider.answer(CAPITAL_FRANCE, language="eng_Latn")
        self.assertIsInstance(resp, GeneralKnowledgeResponse)
        self.assertEqual(resp.answer, "The capital of France is Paris.")
        self.assertEqual(resp.model, "fake-model")
        self.assertEqual(resp.language, "eng_Latn")
        self.assertFalse(resp.abstained)
        self.assertIsNotNone(resp.usage)
        self.assertGreaterEqual(resp.latency_ms, 0)
        model, messages, temperature, max_tokens = backend.provider.calls[0]
        self.assertEqual(model, "fake-model")
        self.assertEqual(temperature, 0.2)
        self.assertEqual(max_tokens, 300)

    def test_general_messages_contain_only_the_user_query(self):
        backend = FakeBackend(FakeChatProvider())
        provider = GeneralKnowledgeProvider(llm=backend)
        provider.answer(CAPITAL_FRANCE, language="eng_Latn")
        _, messages, _, _ = backend.provider.calls[0]
        system, user = messages
        self.assertEqual(system["role"], "system")
        self.assertEqual(user["role"], "user")
        self.assertIn(CAPITAL_FRANCE, user["content"])
        # The critical invariant: no retrieved context is ever included.
        self.assertNotIn("RETRIEVED CONTEXT", user["content"])
        self.assertNotIn("Charles de Gaulle", user["content"])
        self.assertNotIn("[Source", user["content"])

    def test_language_instruction_is_preserved(self):
        backend = FakeBackend(FakeChatProvider(content="پیرس فرانس کا دارالحکومت ہے۔"))
        provider = GeneralKnowledgeProvider(llm=backend)
        resp = provider.answer(URDU_QUERY, language="urd_Arab")
        _, messages, _, _ = backend.provider.calls[0]
        self.assertEqual(resp.language, "urd_Arab")
        self.assertIn("Answer in Urdu", messages[0]["content"])

    def test_provider_error_raises_llm_provider_error(self):
        backend = FakeBackend(FakeChatProvider())
        backend.provider.chat = mock.Mock(side_effect=OSError("backend down"))
        provider = GeneralKnowledgeProvider(llm=backend)
        with self.assertRaises(LLMProviderError):
            provider.answer(CAPITAL_FRANCE)

    def test_plain_dont_know_is_a_normal_answer(self):
        # A generic "I don't know." is a legitimate general-knowledge answer;
        # ``abstained`` only marks the system's explicit RAG abstention.
        backend = FakeBackend(FakeChatProvider(content="I don't know."))
        provider = GeneralKnowledgeProvider(llm=backend)
        resp = provider.answer(CAPITAL_FRANCE)
        self.assertEqual(resp.answer, "I don't know.")
        self.assertFalse(resp.abstained)

    def test_explicit_abstention_message_is_reported(self):
        from app.llm import ABSTENTION_EN

        backend = FakeBackend(FakeChatProvider(content=ABSTENTION_EN))
        provider = GeneralKnowledgeProvider(llm=backend)
        resp = provider.answer(CAPITAL_FRANCE)
        self.assertTrue(resp.abstained)

    def test_provider_name_reflects_backend(self):
        provider = GeneralKnowledgeProvider(llm=FakeBackend(FakeChatProvider()))
        self.assertEqual(provider.provider_name, "fake")


if __name__ == "__main__":
    unittest.main()

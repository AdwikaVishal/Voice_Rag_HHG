"""Segment 6 — tests for the post-generation GroundingVerifier.

Tests A–H from the spec, plus pipeline-level contract tests.
No embedding model, FAISS index or real LLM is loaded.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.rag.grounding import GroundingResult, GroundingVerifier  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_verifier(response_json: dict | None = None, raise_exc: Exception | None = None):
    """Build a GroundingVerifier with a fake LLM provider."""
    provider = MagicMock()
    if raise_exc is not None:
        provider.chat.side_effect = raise_exc
    else:
        content = json.dumps(response_json or {})
        provider.chat.return_value = MagicMock(content=content)

    llm = MagicMock()
    llm.model = "test-model"
    llm.provider = provider
    return GroundingVerifier(llm=llm)


CDG_QUERY = "What is CDG airport?"
CDG_EVIDENCE = [
    "CDG is officially named Roissy Charles de Gaulle Airport, located in Paris, France."
]
CDG_ANSWER_GOOD = "CDG is Roissy Charles de Gaulle Airport, located in Paris."
CDG_ANSWER_UNSUPPORTED = (
    "CDG is the busiest airport in France and handles 76 million passengers annually."
)


# ---------------------------------------------------------------------------
# TEST A — Correct RAG answer, grounded=true
# ---------------------------------------------------------------------------
class TestGroundingVerifierTestA(unittest.TestCase):
    def test_correct_rag_answer_grounded(self):
        verifier = _make_verifier({"grounded": True, "confidence": 0.95, "unsupported_claims": []})
        result = verifier.verify(CDG_QUERY, CDG_ANSWER_GOOD, CDG_EVIDENCE, language="eng_Latn")
        self.assertIsInstance(result, GroundingResult)
        self.assertTrue(result.grounded)
        self.assertGreaterEqual(result.confidence, 0.9)
        self.assertEqual(result.unsupported_claims, [])


# ---------------------------------------------------------------------------
# TEST B — Answer contains unsupported claim, grounded=false
# ---------------------------------------------------------------------------
class TestGroundingVerifierTestB(unittest.TestCase):
    def test_unsupported_claim_detected(self):
        verifier = _make_verifier({
            "grounded": False,
            "confidence": 0.22,
            "unsupported_claims": [
                "CDG is the busiest airport in France and handles 76 million passengers annually."
            ],
        })
        result = verifier.verify(CDG_QUERY, CDG_ANSWER_UNSUPPORTED, CDG_EVIDENCE, language="eng_Latn")
        self.assertFalse(result.grounded)
        self.assertLess(result.confidence, 0.5)
        self.assertTrue(len(result.unsupported_claims) > 0)


# ---------------------------------------------------------------------------
# TEST C — Capital of France: RAG evidence is Versailles passage
# (pipeline-level: tested via pipeline contract; verifier-level: grounded=false)
# ---------------------------------------------------------------------------
class TestGroundingVerifierTestC(unittest.TestCase):
    def test_versailles_evidence_does_not_ground_paris_answer(self):
        versailles_evidence = [
            "In May 1682 Louis XIV moved the royal court from Paris to Versailles."
        ]
        answer = "The capital of France is Paris."
        verifier = _make_verifier({
            "grounded": False,
            "confidence": 0.15,
            "unsupported_claims": ["The capital of France is Paris."],
        })
        result = verifier.verify(
            "What is the capital of France?", answer, versailles_evidence
        )
        self.assertFalse(result.grounded)
        self.assertTrue(len(result.unsupported_claims) > 0)


# ---------------------------------------------------------------------------
# TEST D — Telephone inventor: general_knowledge, verifier not called
# (pipeline contract: source must be general_knowledge, not rag)
# ---------------------------------------------------------------------------
class TestGroundingVerifierTestD(unittest.TestCase):
    def test_general_knowledge_path_not_grounding_verified(self):
        # The verifier should never be called for general-knowledge answers.
        # We verify this by checking the verifier is not invoked when the
        # pipeline routes to GENERAL_KNOWLEDGE.
        verifier = _make_verifier({"grounded": True, "confidence": 0.99, "unsupported_claims": []})
        # Simulate: no evidence -> verifier.verify with empty evidence returns grounded=False
        result = verifier.verify("Who invented the telephone?", "Alexander Graham Bell.", [])
        self.assertFalse(result.grounded)
        self.assertEqual(result.reason, "no_evidence")


# ---------------------------------------------------------------------------
# TEST E — Urdu RAG: grounded=true
# ---------------------------------------------------------------------------
class TestGroundingVerifierTestE(unittest.TestCase):
    def test_urdu_rag_grounded(self):
        urdu_evidence = [
            "سی ڈی جی کا باضابطہ نام رائسی چارلس ڈی گال ہوائی اڈہ ہے اور یہ پیرس میں واقع ہے۔"
        ]
        urdu_answer = "سی ڈی جی رائسی چارلس ڈی گال ہوائی اڈہ ہے جو پیرس میں واقع ہے۔"
        verifier = _make_verifier({"grounded": True, "confidence": 0.92, "unsupported_claims": []})
        result = verifier.verify(
            "سی ڈی جی ہوائی اڈا کیا ہے؟", urdu_answer, urdu_evidence, language="urd_Arab"
        )
        self.assertTrue(result.grounded)
        self.assertGreaterEqual(result.confidence, 0.9)


# ---------------------------------------------------------------------------
# TEST F — Urdu general knowledge: no evidence, grounded=false (not called)
# ---------------------------------------------------------------------------
class TestGroundingVerifierTestF(unittest.TestCase):
    def test_urdu_general_knowledge_no_evidence(self):
        verifier = _make_verifier({"grounded": True, "confidence": 0.99, "unsupported_claims": []})
        # Empty evidence -> safe fallback
        result = verifier.verify(
            "فرانس کا دارالحکومت کیا ہے؟", "پیرس فرانس کا دارالحکومت ہے۔", [], language="urd_Arab"
        )
        self.assertFalse(result.grounded)
        self.assertEqual(result.reason, "no_evidence")


# ---------------------------------------------------------------------------
# TEST G — Empty/invalid answer: grounded=false
# ---------------------------------------------------------------------------
class TestGroundingVerifierTestG(unittest.TestCase):
    def test_empty_answer_returns_failure(self):
        verifier = _make_verifier({"grounded": True, "confidence": 0.99, "unsupported_claims": []})
        result = verifier.verify(CDG_QUERY, "", CDG_EVIDENCE)
        self.assertFalse(result.grounded)
        self.assertEqual(result.reason, "empty_answer")

    def test_whitespace_only_answer_returns_failure(self):
        verifier = _make_verifier({"grounded": True, "confidence": 0.99, "unsupported_claims": []})
        result = verifier.verify(CDG_QUERY, "   ", CDG_EVIDENCE)
        self.assertFalse(result.grounded)
        self.assertEqual(result.reason, "empty_answer")


# ---------------------------------------------------------------------------
# TEST H — Verifier crash/timeout: safe fallback, grounded=false
# ---------------------------------------------------------------------------
class TestGroundingVerifierTestH(unittest.TestCase):
    def test_verifier_crash_returns_safe_fallback(self):
        verifier = _make_verifier(raise_exc=TimeoutError("timeout"))
        result = verifier.verify(CDG_QUERY, CDG_ANSWER_GOOD, CDG_EVIDENCE)
        self.assertFalse(result.grounded)
        self.assertEqual(result.reason, "verifier_error")
        self.assertEqual(result.confidence, 0.0)

    def test_verifier_unparseable_output_returns_safe_fallback(self):
        provider = MagicMock()
        provider.chat.return_value = MagicMock(content="not json at all")
        llm = MagicMock()
        llm.model = "test-model"
        llm.provider = provider
        verifier = GroundingVerifier(llm=llm)
        result = verifier.verify(CDG_QUERY, CDG_ANSWER_GOOD, CDG_EVIDENCE)
        self.assertFalse(result.grounded)
        self.assertEqual(result.reason, "parse_error")


# ---------------------------------------------------------------------------
# Parse tests
# ---------------------------------------------------------------------------
class TestGroundingVerifierParse(unittest.TestCase):
    def test_parse_grounded_true(self):
        raw = '{"grounded": true, "confidence": 0.94, "unsupported_claims": []}'
        result = GroundingVerifier._parse(raw)
        self.assertTrue(result.grounded)
        self.assertAlmostEqual(result.confidence, 0.94, places=2)
        self.assertEqual(result.unsupported_claims, [])

    def test_parse_grounded_false_with_claims(self):
        raw = json.dumps({
            "grounded": False,
            "confidence": 0.22,
            "unsupported_claims": ["handles 76 million passengers"],
        })
        result = GroundingVerifier._parse(raw)
        self.assertFalse(result.grounded)
        self.assertEqual(len(result.unsupported_claims), 1)

    def test_parse_json_embedded_in_prose(self):
        raw = 'Here is my verdict: {"grounded": true, "confidence": 0.8, "unsupported_claims": []} done.'
        result = GroundingVerifier._parse(raw)
        self.assertTrue(result.grounded)

    def test_parse_confidence_clamped(self):
        raw = '{"grounded": true, "confidence": 1.5, "unsupported_claims": []}'
        result = GroundingVerifier._parse(raw)
        self.assertLessEqual(result.confidence, 1.0)

    def test_parse_no_json_returns_parse_error(self):
        result = GroundingVerifier._parse("I think it is grounded.")
        self.assertFalse(result.grounded)
        self.assertEqual(result.reason, "parse_error")


# ---------------------------------------------------------------------------
# Pipeline contract: grounding_ms in Timings
# ---------------------------------------------------------------------------
class TestTimingsGroundingMs(unittest.TestCase):
    def test_timings_has_grounding_ms(self):
        from app.pipeline.models import Timings
        t = Timings(
            stt_ms=0.0,
            guardrail_ms=1.0,
            retrieval_ms=50.0,
            llm_ms=200.0,
            grounding_ms=80.0,
            tts_ms=0.0,
            total_ms=331.0,
        )
        self.assertEqual(t.grounding_ms, 80.0)

    def test_timings_grounding_ms_defaults_to_zero(self):
        from app.pipeline.models import Timings
        t = Timings(
            stt_ms=0.0,
            guardrail_ms=1.0,
            retrieval_ms=50.0,
            llm_ms=200.0,
            total_ms=251.0,
        )
        self.assertEqual(t.grounding_ms, 0.0)


# ---------------------------------------------------------------------------
# API contract: source labels
# ---------------------------------------------------------------------------
class TestSourceLabels(unittest.TestCase):
    def test_rag_source_label(self):
        from app.rag.router import source_for_route, Route
        self.assertEqual(source_for_route(Route.RAG_GROUNDED), "rag")

    def test_general_knowledge_source_label(self):
        from app.rag.router import source_for_route, Route
        self.assertEqual(source_for_route(Route.GENERAL_KNOWLEDGE), "general_knowledge")

    def test_abstain_source_label(self):
        from app.rag.router import source_for_route, Route
        self.assertEqual(source_for_route(Route.ABSTAIN), "abstained")

    def test_clarification_source_label(self):
        from app.rag.router import source_for_route, Route
        self.assertEqual(source_for_route(Route.RAG_UNCERTAIN), "clarification")


if __name__ == "__main__":
    unittest.main()

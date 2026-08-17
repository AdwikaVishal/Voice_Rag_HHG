"""Segment 5.1 — tests for the deterministic answer-support verifier.

The verifier is a pure function of the query and the supporting evidence: a
passage directly answers the question iff it contains the FULL set of the
query's content tokens AND has question-answer structure (a question mark). No
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

from app.rag import AnswerSupportVerifier, SupportVerdict  # noqa: E402
from app.rag.verifier import _fold_token, _question_answer_passage  # noqa: E402

CDG_QUERY = "What is CDG airport?"
CDG_CONTEXT = [
    "Which airport is CDG? CDG is officially named Roissy Charles de Gaulle, "
    "and is located in the city of Paris."
]
VERSAILLES_CONTEXT = [
    "In May 1682 Louis moved the capital of France from Paris to Versailles."
]
TELEPHONE_CONTEXT = [
    "Telephone safety standards establish acoustic pressure limits for handsets."
]
EL_SALVADOR_CONTEXT = [
    "Where is El Salvador? What is the Capital of El Salvador? List of Airports in El Salvador"
]
URDU_QUERY = "سی ڈی جی ہوائی اڈا کیا ہے؟"
URDU_CONTEXT = [
    "سی ڈی جی کون سا ہوائی اڈہ سی ڈی جی ہے؟ سی ڈی جی کا باضابطہ نام رائسی "
    "چارلس ڈی گال ہے اور یہ پیرس میں واقع ہے۔"
]


class TestFoldToken(unittest.TestCase):
    def test_urdu_final_vowel_folding(self):
        self.assertEqual(_fold_token("اڈا"), _fold_token("اڈہ"))
        self.assertEqual(_fold_token("اڈا"), _fold_token("اڈے"))

    def test_ascii_unchanged(self):
        self.assertEqual(_fold_token("cdg"), "cdg")

    def test_empty_token(self):
        self.assertEqual(_fold_token(""), "")


class TestQuestionAnswerPassage(unittest.TestCase):
    def test_ascii_question_mark(self):
        self.assertTrue(_question_answer_passage("Which airport is CDG? CDG is..."))

    def test_urdu_question_mark(self):
        self.assertTrue(_question_answer_passage("سی ڈی جی کون سا ہوائی اڈہ سی ڈی جی ہے؟"))

    def test_no_question_mark(self):
        self.assertFalse(_question_answer_passage("In May 1682 Louis moved the capital."))


class TestAnswerSupportVerifier(unittest.TestCase):
    def test_verify_accepts_question_answer_evidence(self):
        verifier = AnswerSupportVerifier()
        verdict = verifier.verify(CDG_QUERY, CDG_CONTEXT, language="eng_Latn")
        self.assertIsInstance(verdict, SupportVerdict)
        self.assertTrue(verdict.supports_answer)
        self.assertGreaterEqual(verdict.confidence, 0.9)
        self.assertEqual(verdict.reason, "answer_supported")

    def test_verify_accepts_urdu_evidence_with_surface_variant(self):
        verifier = AnswerSupportVerifier()
        verdict = verifier.verify(URDU_QUERY, URDU_CONTEXT, language="urd_Arab")
        self.assertTrue(verdict.supports_answer)
        self.assertEqual(verdict.reason, "answer_supported")

    def test_verify_rejects_historical_narrative_evidence(self):
        verifier = AnswerSupportVerifier()
        verdict = verifier.verify("What is the capital of France?", VERSAILLES_CONTEXT)
        self.assertFalse(verdict.supports_answer)
        self.assertIn(verdict.reason, ("coverage_without_qa_structure", "no_full_answer_coverage"))

    def test_verify_rejects_topic_only_evidence(self):
        verifier = AnswerSupportVerifier()
        verdict = verifier.verify("Who invented the telephone?", TELEPHONE_CONTEXT)
        self.assertFalse(verdict.supports_answer)
        self.assertEqual(verdict.reason, "no_full_answer_coverage")

    def test_verify_rejects_wrong_entity_question(self):
        verifier = AnswerSupportVerifier()
        verdict = verifier.verify("What is the capital of France?", EL_SALVADOR_CONTEXT)
        self.assertFalse(verdict.supports_answer)
        self.assertEqual(verdict.reason, "no_full_answer_coverage")

    def test_verify_full_coverage_without_question_mark_rejects(self):
        # The Versailles narrative contains every content term of the question
        # ("capital", "France") but does not pose or answer it.
        verifier = AnswerSupportVerifier()
        verdict = verifier.verify("What is the capital of France?", VERSAILLES_CONTEXT)
        self.assertFalse(verdict.supports_answer)
        self.assertEqual(verdict.reason, "coverage_without_qa_structure")

    def test_verify_empty_context_rejected(self):
        verifier = AnswerSupportVerifier()
        verdict = verifier.verify("What is the capital of France?", [])
        self.assertFalse(verdict.supports_answer)
        self.assertEqual(verdict.reason, "no_context")
        self.assertEqual(verdict.confidence, 0.0)

    def test_verify_blank_entries_filtered_but_real_evidence_kept(self):
        verifier = AnswerSupportVerifier()
        verdict = verifier.verify(CDG_QUERY, ["   ", None, CDG_CONTEXT[0]])
        self.assertTrue(verdict.supports_answer)

    def test_verify_no_query_content_terms(self):
        verifier = AnswerSupportVerifier()
        verdict = verifier.verify("the and or", CDG_CONTEXT)
        self.assertFalse(verdict.supports_answer)
        self.assertEqual(verdict.reason, "no_query_terms")

    def test_verify_rejection_confidence_in_range(self):
        verifier = AnswerSupportVerifier()
        for query, context in (
            ("What is the capital of France?", VERSAILLES_CONTEXT),
            ("Who invented the telephone?", TELEPHONE_CONTEXT),
        ):
            verdict = verifier.verify(query, context)
            self.assertGreaterEqual(verdict.confidence, 0.0)
            self.assertLessEqual(verdict.confidence, 1.0)

    def test_provider_name_is_deterministic(self):
        verifier = AnswerSupportVerifier()
        self.assertEqual(verifier.provider_name, "deterministic")

    def test_enabled_flag_follows_config(self):
        verifier = AnswerSupportVerifier()
        self.assertTrue(verifier.enabled)
        with mock.patch("app.config.ANSWER_SUPPORT_VERIFY_ENABLED", False):
            self.assertFalse(verifier.enabled)

    def test_language_is_ignored_by_deterministic_check(self):
        verifier = AnswerSupportVerifier()
        with_en = verifier.verify(URDU_QUERY, URDU_CONTEXT, language="eng_Latn")
        with_ur = verifier.verify(URDU_QUERY, URDU_CONTEXT, language="urd_Arab")
        self.assertEqual(with_en.supports_answer, with_ur.supports_answer)


if __name__ == "__main__":
    unittest.main()

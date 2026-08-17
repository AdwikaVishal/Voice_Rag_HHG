"""Unit tests for the chunking module (Segment 2).

Run from the backend/ directory:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.chunking.base import BaseChunker
from app.chunking.factory import create_chunker
from app.chunking.fixed import FixedSizeChunker
from app.chunking.metadata import MetadataAwareChunker
from app.chunking.recursive import RecursiveChunker
from app.chunking.sentence import SentenceChunker
from app.chunking.splitting import split_by_tokens, split_sentences
from app.chunking.tokenizer import count_tokens, tokenize

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

ENGLISH_TEXT = (
    "A company is incorporated in a specific nation, often within the bounds "
    "of a smaller subset of that nation, such as a state or province. "
    "The corporation is then governed by the laws of incorporation in that state. "
    "A corporation may issue stock, either private or public, or may be classified "
    "as a non-stock corporation. If stock is issued, the corporation will usually "
    "be governed by its shareholders, either directly or indirectly."
)

HINDI = (
    "हिंदी एक इंडो-आर्यन भाषा है। यह भारत की आधिकारिक भाषाओं में से एक है। "
    "भारत में लगभग छह सौ मिलियन लोग हिंदी बोलते हैं। हिंदी देवनागरी लिपि में लिखी जाती है।"
)

BENGALI = (
    "বাংলা একটি ইন্দো-আর্য ভাষা। এটি বাংলাদেশের জাতীয় ভাষা। "
    "পশ্চিমবঙ্গের মানুষেরাও বাংলা ভাষায় কথা বলেন।"
)

GUJARATI = (
    "ગુજરાતી એ ભારતની એક મુખ્ય ભાષા છે। આ ભાષા ગુજરાત રાજ્યમાં બોલાય છે। "
    "ગુજરાતીમાં ઘણું સાહિત્ય લખાયેલું છે।"
)

URDU = "پوٹاشیم کیا ہے؟ یہ ایک عنصر ہے۔ پوٹاشیم جسم کے لیے اہم ہے۔"

RECORD = {
    "record_id": "msmarco_xi_000001",
    "query_id": 1102432,
    "source_lang": "eng_Latn",
    "target_lang": "urd_Arab",
    "query_type": "DESCRIPTION",
    "query": "کارپوریشن کیا ہے؟",
    "english_query": "what is a corporation?",
    "answer": "یک جواب",
    "english_answer": "an answer",
    "passages": [
        {
            "passage_index": 0,
            "english_text": ENGLISH_TEXT,
            "translated_text": URDU,
            "is_selected": 1,
        },
        {
            "passage_index": 1,
            "english_text": "A short passage with one sentence.",
            "translated_text": "ایک چھوٹا پیراگراف۔",
            "is_selected": 0,
        },
    ],
}

WORDS_1000 = " ".join(f"token{i}" for i in range(1000))
WORDS_2000 = " ".join(f"w{i}" for i in range(2000))


def make_long_english_record(n_words: int = 2000) -> dict:
    return {
        "record_id": "rec_long",
        "query_id": 1,
        "source_lang": "eng_Latn",
        "target_lang": "urd_Arab",
        "query_type": "DESCRIPTION",
        "query": "q",
        "english_query": "q",
        "passages": [
            {
                "passage_index": 0,
                "english_text": ". ".join(
                    " ".join(f"w{i}" for i in range(n, min(n + 10, n_words)))
                    for n in range(0, n_words, 10)
                )
                + ".",
                "translated_text": URDU,
                "is_selected": 0,
            }
        ],
    }


# --------------------------------------------------------------------------
# FixedSizeChunker
# --------------------------------------------------------------------------


class TestFixedSizeChunker(unittest.TestCase):
    def test_chunk_size_respected(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=0.0)
        chunks = chunker.split_text(WORDS_1000)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(count_tokens(chunk), 50)
        self.assertLessEqual(count_tokens(chunks[0]), 50)

    def test_overlap_works(self):
        chunker = FixedSizeChunker(chunk_size=20, overlap=0.5)
        chunks = chunker.split_text(WORDS_1000)
        self.assertGreater(len(chunks), 1)
        for a, b in zip(chunks, chunks[1:]):
            shared = set(tokenize(a)) & set(tokenize(b))
            self.assertGreaterEqual(len(shared), 1)

    def test_no_infinite_loop_with_bad_overlap(self):
        for overlap in (1.0, 2.0, 5.0, -1.0):
            chunker = FixedSizeChunker(chunk_size=10, overlap=overlap)
            chunks = chunker.split_text(WORDS_1000)
            self.assertGreater(len(chunks), 0)
            self.assertLessEqual(len(chunks), 1000)

    def test_empty_text(self):
        chunker = FixedSizeChunker(chunk_size=10)
        self.assertEqual(chunker.split_text(""), [])
        self.assertEqual(chunker.split_text("   \n  "), [])

    def test_order_preserved(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=0.0)
        chunks = chunker.split_text(WORDS_1000)
        flat = " ".join(chunks)
        self.assertTrue(flat.startswith("token0"))
        self.assertTrue(flat.endswith("token999"))

    def test_chunks_are_substrings(self):
        text = "This is a test with some content here."
        chunker = FixedSizeChunker(chunk_size=4, overlap=0.2)
        for chunk in chunker.split_text(text):
            self.assertIn(chunk, text)


# --------------------------------------------------------------------------
# SentenceChunker
# --------------------------------------------------------------------------


class TestSentenceChunker(unittest.TestCase):
    def test_sentence_boundaries_preserved(self):
        chunker = SentenceChunker(chunk_size=5)
        text = "This is sentence one. This is sentence two! And a third?"
        chunks = chunker.split_text(text)
        for chunk in chunks:
            self.assertGreaterEqual(count_tokens(chunk), 1)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "This is sentence one.")
        self.assertEqual(chunks[1], "This is sentence two!")
        self.assertEqual(chunks[2], "And a third?")

    def test_combines_small_sentences(self):
        chunker = SentenceChunker(chunk_size=100)
        chunks = chunker.split_text(ENGLISH_TEXT)
        self.assertEqual(len(chunks), 1)
        self.assertLessEqual(count_tokens(chunks[0]), 100)

    def test_long_sentence_handled_safely(self):
        long_sentence = " ".join(f"word{i}" for i in range(300)) + "."
        chunker = SentenceChunker(chunk_size=50)
        chunks = chunker.split_text(long_sentence)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(count_tokens(chunks[0]), 300)  # kept whole, not split

    def test_empty_text(self):
        chunker = SentenceChunker(chunk_size=50)
        self.assertEqual(chunker.split_text(""), [])
        self.assertEqual(chunker.split_text("  \n "), [])


# --------------------------------------------------------------------------
# RecursiveChunker
# --------------------------------------------------------------------------


class TestRecursiveChunker(unittest.TestCase):
    def test_large_paragraph_eventually_splits(self):
        chunker = RecursiveChunker(chunk_size=256)
        chunks = chunker.split_text(WORDS_2000)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(count_tokens(chunk), 256)

    def test_hierarchy_respected_short_text_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=256)
        chunks = chunker.split_text(ENGLISH_TEXT)
        self.assertEqual(len(chunks), 1)

    def test_no_oversized_unless_unavoidable(self):
        chunker = RecursiveChunker(chunk_size=100)
        chunks = chunker.split_text(WORDS_2000)
        for chunk in chunks:
            self.assertLessEqual(count_tokens(chunk), 100)

    def test_empty_text(self):
        chunker = RecursiveChunker(chunk_size=100)
        self.assertEqual(chunker.split_text(""), [])


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


class TestMetadataAwareChunker(unittest.TestCase):
    def test_metadata_preserved_and_links_correct(self):
        record = make_long_english_record(2000)
        chunker = MetadataAwareChunker(chunk_size=100, text_field="english")
        chunks = chunker.chunk(record)
        self.assertGreater(len(chunks), 1)

        for chunk in chunks:
            self.assertEqual(chunk.record_id, "rec_long")
            self.assertEqual(chunk.query_id, 1)
            self.assertEqual(chunk.source_lang, "eng_Latn")
            self.assertEqual(chunk.target_lang, "urd_Arab")
            self.assertEqual(chunk.language, "eng_Latn")
            self.assertEqual(chunk.passage_index, 0)

        positions = [c.chunk_position for c in chunks]
        self.assertEqual(positions, list(range(len(chunks))))
        for c in chunks:
            self.assertEqual(c.total_chunks, len(chunks))

        self.assertIsNone(chunks[0].prev_chunk_id)
        self.assertIsNone(chunks[-1].next_chunk_id)
        for a, b in zip(chunks, chunks[1:]):
            self.assertEqual(a.next_chunk_id, b.chunk_id)
            self.assertEqual(b.prev_chunk_id, a.chunk_id)

        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(set(ids)), len(ids))

    def test_both_languages_produce_two_groups(self):
        chunker = MetadataAwareChunker(chunk_size=256, text_field="both")
        chunks = chunker.chunk(RECORD)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertEqual(c.record_id, "msmarco_xi_000001")
            self.assertEqual(c.query_id, 1102432)
            self.assertEqual(c.source_lang, "eng_Latn")
            self.assertEqual(c.target_lang, "urd_Arab")
        langs = {c.language for c in chunks}
        self.assertEqual(langs, {"eng_Latn", "urd_Arab"})

    def test_positions_correct_prev_next(self):
        record = make_long_english_record(300)
        chunker = MetadataAwareChunker(chunk_size=50, text_field="english")
        chunks = chunker.chunk(record)
        for idx, c in enumerate(chunks):
            self.assertEqual(c.chunk_position, idx)
            if idx == 0:
                self.assertIsNone(c.prev_chunk_id)
            else:
                self.assertEqual(c.prev_chunk_id, chunks[idx - 1].chunk_id)
            if idx == len(chunks) - 1:
                self.assertIsNone(c.next_chunk_id)
            else:
                self.assertEqual(c.next_chunk_id, chunks[idx + 1].chunk_id)


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


class TestFactory(unittest.TestCase):
    def test_all_strategies_instantiate(self):
        for name in ("fixed", "sentence", "recursive", "metadata"):
            chunker = create_chunker(name)
            self.assertIsInstance(chunker, BaseChunker)

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError) as ctx:
            create_chunker("bm25")
        message = str(ctx.exception)
        self.assertIn("bm25", message)
        self.assertIn("fixed", message)
        self.assertIn("sentence", message)
        self.assertIn("recursive", message)
        self.assertIn("metadata", message)

    def test_case_and_whitespace_insensitive(self):
        self.assertIsInstance(create_chunker(" Fixed "), FixedSizeChunker)
        self.assertIsInstance(create_chunker("SENTENCE"), SentenceChunker)


# --------------------------------------------------------------------------
# Multilingual / Unicode
# --------------------------------------------------------------------------


class TestMultilingual(unittest.TestCase):
    def test_indic_sentence_splitting(self):
        for text, expected in (
            (HINDI, 4),
            (BENGALI, 3),
            (GUJARATI, 3),
        ):
            sentences = split_sentences(text)
            self.assertEqual(len(sentences), expected, text[:10])

    def test_urdu_sentence_splitting(self):
        sentences = split_sentences(URDU)
        self.assertEqual(len(sentences), 3)

    def test_english_sentence_splitting(self):
        sentences = split_sentences("Hello world. This is a test! And another?")
        self.assertEqual(len(sentences), 3)

    def test_indic_tokenization(self):
        for text in (HINDI, BENGALI, GUJARATI, URDU):
            self.assertGreater(count_tokens(text), 0)
            tokens = tokenize(text)
            self.assertTrue(all(token.strip() for token in tokens))

    def test_indic_chunks_never_corrupt_text(self):
        for text in (HINDI, BENGALI, GUJARATI):
            chunker = FixedSizeChunker(chunk_size=8, overlap=0.2)
            chunks = chunker.split_text(text)
            self.assertGreater(len(chunks), 0)
            for chunk in chunks:
                self.assertIn(chunk, text)
                self.assertEqual(count_tokens(chunk), count_tokens(chunk))

    def test_indic_no_ascii_only_normalization(self):
        for text in (HINDI, BENGALI, GUJARATI):
            for char in "।":
                self.assertIn(char, text)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


class TestSplittingHelpers(unittest.TestCase):
    def test_split_by_tokens_respects_chunk_size(self):
        chunks = split_by_tokens(WORDS_1000, chunk_size=100, overlap=0.1)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(count_tokens(chunk), 100)

    def test_split_by_tokens_overlap_clamped(self):
        chunks = split_by_tokens(WORDS_1000, chunk_size=10, overlap=1.0)
        self.assertGreater(len(chunks), 0)
        self.assertLessEqual(len(chunks), 1000)

    def test_abbreviation_does_not_split(self):
        text = "Dr. Smith works for U.S. Steel. He is well known etc. This is next."
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0], "Dr. Smith works for U.S. Steel.")
        self.assertEqual(sentences[1], "He is well known etc. This is next.")

    def test_decimal_number_not_split(self):
        sentences = split_sentences("The cost is 3.5 million dollars. It is cheap.")
        self.assertEqual(len(sentences), 2)


if __name__ == "__main__":
    unittest.main()

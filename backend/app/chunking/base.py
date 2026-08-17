"""Abstract chunker and the record -> chunks enrichment pipeline.

Subclasses only implement :meth:`BaseChunker.split_text` (a pure
``text -> list[str]`` function). The base class handles record traversal,
metadata preservation, chunk ids and prev/next linking uniformly, so every
strategy produces the same typed :class:`Chunk` model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Chunk
from .tokenizer import count_tokens

_VALID_TEXT_FIELDS = ("english", "translated", "both")


class BaseChunker(ABC):
    """Base class for all chunking strategies."""

    strategy_name = "base"

    def __init__(self, chunk_size: int = 256, overlap: float = 0.20, text_field: str = "both"):
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if text_field not in _VALID_TEXT_FIELDS:
            raise ValueError(
                f"text_field must be one of {_VALID_TEXT_FIELDS}, got {text_field!r}"
            )
        self.chunk_size = int(chunk_size)
        self.overlap = float(overlap)
        self.text_field = text_field

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split a single text unit into chunk pieces.

        Must return a list of non-empty strings, preserving order.
        """

    # ------------------------------------------------------------------
    def chunk(self, record: dict) -> list[Chunk]:
        """Chunk every passage text unit of ``record`` into :class:`Chunk` s."""
        record_id = record.get("record_id")
        if not record_id:
            raise ValueError("record is missing a non-empty 'record_id'")

        chunks: list[Chunk] = []
        for passage in record.get("passages") or []:
            passage_index = passage.get("passage_index", 0)
            is_selected = int(passage.get("is_selected") or 0)
            for field in self._text_fields():
                text = passage.get(field)
                if not text or not text.strip():
                    continue
                language = self._language_for(record, field)
                pieces = self.split_text(text)
                chunks.extend(
                    self._materialize(
                        record, passage_index, is_selected, field, language, pieces
                    )
                )
        return chunks

    def _text_fields(self) -> list[str]:
        if self.text_field == "both":
            return ["english_text", "translated_text"]
        return [f"{self.text_field}_text"]

    def _language_for(self, record: dict, field: str) -> str:
        if field == "english_text":
            return record.get("source_lang") or record.get("target_lang") or ""
        return record.get("target_lang") or record.get("source_lang") or ""

    def _materialize(
        self,
        record: dict,
        passage_index: int,
        is_selected: int,
        field: str,
        language: str,
        pieces: list[str],
    ) -> list[Chunk]:
        total = len(pieces)
        unit: list[Chunk] = []
        lang_tag = language if language else field
        for pos, text in enumerate(pieces):
            unit.append(
                Chunk(
                    chunk_id=f"{record['record_id']}::{passage_index}::{lang_tag}::{pos}",
                    record_id=record["record_id"],
                    query_id=record.get("query_id"),
                    text=text,
                    language=language,
                    chunk_position=pos,
                    total_chunks=total,
                    prev_chunk_id=None,
                    next_chunk_id=None,
                    source_lang=record.get("source_lang"),
                    target_lang=record.get("target_lang"),
                    query_type=record.get("query_type"),
                    query=record.get("query") or "",
                    english_query=record.get("english_query") or "",
                    passage_index=passage_index,
                    is_selected=is_selected,
                    text_field=field,
                    char_count=len(text),
                    token_count=count_tokens(text),
                )
            )
        for idx in range(len(unit)):
            if idx > 0:
                unit[idx].prev_chunk_id = unit[idx - 1].chunk_id
            if idx < total - 1:
                unit[idx].next_chunk_id = unit[idx + 1].chunk_id
        return unit

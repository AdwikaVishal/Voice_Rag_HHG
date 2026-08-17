"""Typed models for the chunking module."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Chunk(BaseModel):
    """A single retrieval-ready text chunk with preserved source metadata.

    ``prev_chunk_id`` / ``next_chunk_id`` link each chunk to its neighbours
    within the same passage + language group, so a retrieved chunk can later
    be expanded with its context.

    All metadata values come from the source record; nothing is invented.
    """

    chunk_id: str
    record_id: str
    query_id: Optional[int] = None
    text: str
    language: str = ""
    chunk_position: int = 0
    total_chunks: int = 0
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    # --- preserved source metadata ----------------------------------------
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    query_type: Optional[str] = None
    query: str = ""
    english_query: str = ""

    # --- passage-level metadata -------------------------------------------
    passage_index: int = 0
    is_selected: int = 0
    text_field: str = ""  # which passage field produced this chunk
    char_count: int = 0
    token_count: int = 0

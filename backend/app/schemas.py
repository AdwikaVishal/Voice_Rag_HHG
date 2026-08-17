"""Pydantic request/response schemas for the retrieval API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from .config import DEFAULT_TOP_K, MAX_TOP_K


class SearchRequest(BaseModel):
    """Body of ``POST /search``."""

    query: str = Field(..., min_length=1, description="Natural-language query (English or Urdu).")
    top_k: int = Field(
        DEFAULT_TOP_K,
        ge=1,
        le=MAX_TOP_K,
        description=f"Number of chunks to return (1..{MAX_TOP_K}).",
    )
    language: Optional[str] = Field(
        None,
        description="Optional language filter (eng_Latn | urd_Arab); auto-detected when omitted.",
    )

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class RetrievedChunk(BaseModel):
    """One retrieved chunk, with the identifiers downstream stages need."""

    chunk_id: str
    score: float
    text: str
    record_id: str
    query_id: int
    passage_index: int
    language: str
    chunk_position: Optional[int] = None
    total_chunks: Optional[int] = None
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    is_selected: Optional[int] = None
    selected_by_hybrid: bool = False
    metadata: dict[str, Any] = {}

    @classmethod
    def from_result(cls, result: Any) -> "RetrievedChunk":
        """Build a chunk schema from a :class:`RetrievalResult`."""
        metadata = result.metadata or {}
        return cls(
            chunk_id=result.chunk_id,
            score=round(float(result.score), 5),
            text=result.text,
            record_id=str(metadata.get("record_id", "")),
            query_id=int(metadata.get("query_id", 0)),
            passage_index=int(metadata.get("passage_index", 0)),
            language=str(metadata.get("language", "")),
            chunk_position=metadata.get("chunk_position"),
            total_chunks=metadata.get("total_chunks"),
            prev_chunk_id=metadata.get("prev_chunk_id"),
            next_chunk_id=metadata.get("next_chunk_id"),
            is_selected=metadata.get("is_selected"),
            selected_by_hybrid=bool(getattr(result, "selected_by_hybrid", False)),
            metadata=metadata,
        )


class SearchResponse(BaseModel):
    """Response of ``POST /search``."""

    query: str
    detected_language: Optional[str]
    top_k: int
    strategy: str
    index_chunks: int
    latency_ms: float
    results: list[RetrievedChunk]

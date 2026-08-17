"""Segment 5 — routing / fallback behavior against the FULL recursive index.

Runs the real production text path (guardrail -> hybrid retrieval ->
answerability gate -> router -> answer) through :class:`QueryPipeline` for the
evidence-present CDG queries and the known coverage-gap questions.

Expected on the full index:
  * CDG EN / CDG Urdu      -> RAG_GROUNDED  (source=rag)
  * "What is the capital of France?"  -> GENERAL_KNOWLEDGE (source=general_knowledge)
  * "Who invented the telephone?"     -> GENERAL_KNOWLEDGE (source=general_knowledge)
  * "فرانس کا دارالحکومت کیا ہے؟"      -> GENERAL_KNOWLEDGE (source=general_knowledge)

No writes; requires the full index and a running LLM backend (Ollama by
default) for the answer legs.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.pipeline import QueryPipeline  # noqa: E402

QUERIES = [
    ("What is CDG airport?", "eng_Latn"),
    ("What is the capital of France?", "eng_Latn"),
    ("Who invented the telephone?", "eng_Latn"),
    ("فرانس کا دارالحکومت کیا ہے؟", "urd_Arab"),
    ("سی ڈی جی ہوائی اڈا کیا ہے؟", "urd_Arab"),
]


def main() -> None:
    pipeline = QueryPipeline()
    print(f"Pipeline ready (index strategy={pipeline.retriever.strategy})\n")
    for query, lang in QUERIES:
        resp = pipeline.process_text(query, language_hint=lang, top_k=10, with_tts=False)
        gen = resp.generation
        print(f"QUERY: {query!r}  (language={resp.language})")
        print(f"  route     : {gen.source}")
        print(f"  status    : {gen.status}")
        print(f"  confidence: {gen.confidence}")
        print(f"  reason    : {gen.reason}")
        print(f"  evidence  : {gen.evidence_count} chunks, "
              f"supporting={gen.supporting_chunk_ids[:3]}")
        print(f"  sources   : {len(gen.sources)} cited")
        print(f"  answer    : {gen.answer[:200]!r}")
        print()


if __name__ == "__main__":
    main()

"""Fallback router (Segment 5): picks the answer route from the Segment 4
answerability verdict.

The router is a pure, deterministic function of the gate decision — it never
inspects raw retrieval scores and never decides on "results were returned".
Routes:

* ``RAG_GROUNDED``      — gate verdict ANSWERABLE (strong RAG evidence)
* ``RAG_UNCERTAIN``     — gate verdict UNCERTAIN (cautious / clarifying RAG answer)
* ``GENERAL_KNOWLEDGE`` — gate verdict UNANSWERABLE_FROM_RAG and the fallback is
  enabled (the user query is answered by general knowledge, never from the
  irrelevant retrieved chunks)
* ``ABSTAIN``           — gate verdict UNANSWERABLE_FROM_RAG and the fallback is
  disabled (Segment 4 pure abstention)

Each route maps to a stable ``source`` label exposed on the API:
``rag`` / ``clarification`` / ``general_knowledge`` / ``abstained``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ..config import GENERAL_KNOWLEDGE_FALLBACK
from .models import AnswerabilityDecision, AnswerabilityStatus

__all__ = [
    "Route",
    "FallbackRouter",
    "route",
    "source_for_route",
    "get_fallback_router",
]


class Route:
    """Machine-readable answer routes (kept separate from the API status)."""

    RAG_GROUNDED = "RAG_GROUNDED"
    RAG_UNCERTAIN = "RAG_UNCERTAIN"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    ABSTAIN = "ABSTAIN"


# Route -> stable API ``source`` label.
_ROUTE_SOURCE = {
    Route.RAG_GROUNDED: "rag",
    Route.RAG_UNCERTAIN: "clarification",
    Route.GENERAL_KNOWLEDGE: "general_knowledge",
    Route.ABSTAIN: "abstained",
}


def route(
    decision: AnswerabilityDecision,
    enable_general_knowledge: bool = True,
) -> str:
    """Map an answerability verdict to an answer route.

    ``enable_general_knowledge`` controls whether an
    ``UNANSWERABLE_FROM_RAG`` verdict routes to the general-knowledge provider
    (``GENERAL_KNOWLEDGE``) or to the Segment 4 pure abstention (``ABSTAIN``).
    """
    if decision.status == AnswerabilityStatus.ANSWERABLE:
        return Route.RAG_GROUNDED
    if decision.status == AnswerabilityStatus.UNCERTAIN:
        return Route.RAG_UNCERTAIN
    # UNANSWERABLE_FROM_RAG — the gate found no usable evidence.
    if enable_general_knowledge:
        return Route.GENERAL_KNOWLEDGE
    return Route.ABSTAIN


def source_for_route(selected_route: str) -> str:
    """Stable API ``source`` label for a route (unknown routes -> ``rag``)."""
    return _ROUTE_SOURCE.get(selected_route, _ROUTE_SOURCE[Route.RAG_GROUNDED])


class FallbackRouter:
    """Configurable router; holds the general-knowledge enable flag."""

    def __init__(self, enable_general_knowledge: Optional[bool] = None) -> None:
        self.enable_general_knowledge = (
            GENERAL_KNOWLEDGE_FALLBACK
            if enable_general_knowledge is None
            else bool(enable_general_knowledge)
        )

    def route(self, decision: AnswerabilityDecision) -> str:
        """Pick the answer route for one gate verdict."""
        return route(decision, enable_general_knowledge=self.enable_general_knowledge)


@lru_cache(maxsize=1)
def get_fallback_router() -> FallbackRouter:
    """Process-wide singleton router (reads config defaults)."""
    return FallbackRouter()

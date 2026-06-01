from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from core.constants import MAX_LOOP_COUNT, MIN_CONFIDENCE_SCORE
from core.nodes import (
    evaluator_node,
    finalizer_node,
    human_review_node,
    planner_node,
    researcher_node,
    synthesizer_node,
)
from core.state import ResearchState

logger = logging.getLogger(__name__)


# ── Routing function ──────────────────────────────────────────────────────────

def _route_after_evaluator(state: ResearchState) -> str:
    """
    Conditional edge: reads confidence_score and loop_count directly from state.
    Returns the node name LangGraph should route to next.
    """
    score = state.get("confidence_score", 0.0)
    loops = state.get("loop_count", 0)

    if score < MIN_CONFIDENCE_SCORE and loops < MAX_LOOP_COUNT:
        logger.info(
            "Routing → retry (score=%.2f, loop=%d)", score, loops
        )
        return "researcher"

    logger.info(
        "Routing → finalize (score=%.2f, loop=%d)", score, loops
    )
    return "finalizer"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(checkpointer: object | None = None) -> StateGraph:
    """
    Builds and compiles the research graph.

    Args:
        checkpointer: Persistence backend for HITL and state snapshots.
                      Default: MemorySaver (in-memory, for development).
                      Production: SqliteSaver or PostgresSaver.
    """
    builder = StateGraph(ResearchState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("planner",      planner_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("researcher",   researcher_node)
    builder.add_node("synthesizer",  synthesizer_node)
    builder.add_node("evaluator",    evaluator_node)
    builder.add_node("finalizer",    finalizer_node)

    # ── Deterministic edges ───────────────────────────────────────────────────
    builder.add_edge(START,          "planner")
    builder.add_edge("planner",      "human_review")
    builder.add_edge("human_review", "researcher")
    builder.add_edge("researcher",   "synthesizer")
    builder.add_edge("synthesizer",  "evaluator")
    builder.add_edge("finalizer",    END)

    # ── Conditional edge from evaluator ───────────────────────────────────────
    builder.add_conditional_edges(
        source="evaluator",
        path=_route_after_evaluator,
        path_map={
            "researcher": "researcher",
            "finalizer":  "finalizer",
        },
    )

    return builder.compile(
        checkpointer=checkpointer or MemorySaver(),
    )


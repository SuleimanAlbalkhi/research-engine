from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage, HumanMessage

from core.config import get_settings
from core.constants import MAX_LOOP_COUNT, MIN_CONFIDENCE_SCORE
from core.providers import get_llm, get_search_tool
from core.state import ResearchState
from schemas.output_schemas import (
    SearchQueriesOutput,
    SynthesizerOutput as SynthesisOutput,
    EvaluatorOutput as EvaluationOutput,
)

logger = logging.getLogger(__name__)


def planner_node(state: ResearchState) -> dict:
    """
    Generates structured search queries from the user configuration.

    Reads:  topic, persona, search_depth
    Writes: search_queries
    """
    settings = get_settings()
    structured_llm = get_llm().with_structured_output(SearchQueriesOutput)

    query_count = _queries_for_depth(state["search_depth"])

    messages = [
        SystemMessage(content=(
            f"You are a precise research planner. "
            f"Your task is to create search queries for a {state['persona']}. "
            f"Think step by step: What do you need to know about the topic? "
            f"Which aspects are particularly relevant for a {state['persona']}?"
        )),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Persona: {state['persona']}\n"
            f"Search Depth: {state['search_depth']}/5\n\n"
            f"Generate exactly {query_count} precise, different search queries in English. "
            f"Every query must cover a different aspect of the topic."
        )),
    ]

    try:
        output: SearchQueriesOutput = structured_llm.invoke(messages)
        logger.info(
            "Planner generated %d queries for topic '%s'",
            len(output.queries),
            state["topic"],
        )
        return {"search_queries": output.queries}

    except Exception as e:
        logger.error("Planner-Node failed: %s", e)
        raise


def _queries_for_depth(depth: int) -> int:
    """translates search_depth (1–5) in a concrete Query count."""
    mapping = {1: 2, 2: 3, 3: 5, 4: 7, 5: 10}
    return mapping.get(depth, 5)

from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def researcher_node(state: ResearchState) -> dict:
    """
    Runs validated search queries in parallel and collects raw data.

    Reads:   validated_search_queries
    Writes: retrieved_docs  ← Reducer accumulates over loop iterations
    """
    queries = state["validated_search_queries"]
    search_tool = get_search_tool()

    logger.info(
        "Researcher started %d parallel queries (Loop %d)",
        len(queries),
        state.get("loop_count", 0),
    )

    def _search_one(query: str) -> dict:
        """Runs a single search. Errors are isolated – never propagated."""
        start = time.perf_counter()
        try:
            content = search_tool.invoke(query)
            return {
                "query": query,
                "content": content,
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "error": None,
            }
        except Exception as e:
            logger.warning("Query failed: '%s' → %s", query, e)
            return {
                "query": query,
                "content": "",
                "duration_ms": None,
                "error": str(e),
            }

    results: list[dict] = []

    # max_workers=5: suitable for 2–10 queries; more threads offer
    # no benefit for I/O-bound tasks and only increase overhead.
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_search_one, q): q for q in queries}
        for future in as_completed(futures):
            results.append(future.result())

    valid_docs = [r for r in results if r["error"] is None and r["content"]]

    if not valid_docs:
        raise RuntimeError(
            f"All {len(queries)} queries failed. "
            "Check your network connection or switch the search provider."
        )

    logger.info(
        "%d/%d queries successful. Reducer appends docs to existing ones.",
        len(valid_docs),
        len(queries),
    )

    # No overwriting – the reducer in ResearchState handles the append.
    return {"retrieved_docs": valid_docs}



def synthesizer_node(state: ResearchState) -> dict:
    """
    Synthesizes retrieved_docs into a structured report draft.

    Reads:   retrieved_docs, topic, persona, output_format, feedback (optional)
    Writes:  draft (Markdown string), sources (List[dict])
    """
    structured_llm = get_llm().with_structured_output(SynthesisOutput)

    is_retry = bool(state.get("feedback")) and state.get("loop_count", 0) > 0
    logger.info(
        "Synthesizer started (Loop %d, retry=%s)",
        state.get("loop_count", 0),
        is_retry,
    )

    docs_text = _format_docs_for_prompt(state.get("retrieved_docs", []))

    feedback_block = ""
    if is_retry:
        feedback_block = (
            f"\n\n## IMPORTANT: Improvement Request (previous loop)\n"
            f"{state['feedback']}\n"
            f"Address these points explicitly in your new version.\n"
        )

    messages = [
        SystemMessage(content=(
            f"You are a precise research analyst and author. "
            f"You create a {state['persona']} on the given topic. "
            f"Your output format is: {state['output_format']}.\n\n"
            f"Work through these steps:\n"
            f"1. Analyze the provided sources for key statements\n"
            f"2. Identify 3–5 thematic focal points\n"
            f"3. Assign each source to the appropriate focal points\n"
            f"4. Write the sections based on this structure\n"
            f"5. Derive 3–5 key_findings as concise sentences"
        )),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"{feedback_block}\n"
            f"## Researched Sources\n\n{docs_text}"
        )),
    ]

    try:
        output: SynthesisOutput = structured_llm.invoke(messages)

        draft = _build_draft_markdown(
            topic=state["topic"],
            persona=state["persona"],
            output=output,
        )

        sources = [
            {
                "url": src.url,
                "title": src.title,
                "snippet": src.snippet,
                "relevance_score": src.relevance_score,
            }
            for src in output.sources
        ]

        logger.info(
            "Synthesizer done: %d sections, %d sources",
            len(output.sections),
            len(sources),
        )
        return {"draft": draft, "sources": sources}

    except Exception as e:
        logger.error("Synthesizer node failed: %s", e)
        raise


# Private helper functions

def _format_docs_for_prompt(docs: list[dict], max_chars: int = 12_000) -> str:
    """
    Formats retrieved_docs as readable text for the prompt.
    Truncates to max_chars to preserve context window of local models.
    """
    lines: list[str] = []
    total = 0

    for i, doc in enumerate(reversed(docs), start=1):
        block = (
            f"[{i}] Query: {doc.get('query', 'N/A')}\n"
            f"    Content: {doc.get('content', '')}\n"
        )
        if total + len(block) > max_chars:
            lines.append(f"[... {len(docs) - i + 1} more sources truncated]")
            break
        lines.append(block)
        total += len(block)

    return "\n".join(lines)


def _build_draft_markdown(
    topic: str,
    persona: str,
    output: SynthesisOutput,
) -> str:
    """Converts SynthesisOutput into a readable Markdown string."""
    parts: list[str] = [
        f"# {topic}",
        f"*{persona}*\n",
        "## Key Findings\n",
    ]

    for finding in output.key_findings:
        parts.append(f"- {finding}")

    parts.append("")

    for section in output.sections:
        parts.append(f"## {section.title}\n")
        parts.append(section.content)

        if section.source_urls:
            refs = ", ".join(f"[{url}]" for url in section.source_urls)
            parts.append(f"\n*Sources: {refs}*")

        parts.append("")

    return "\n".join(parts)

def evaluator_node(state: ResearchState) -> dict:
    """
    Evaluates draft quality and decides between retry or finalization.

    Reads:   draft, sources, topic, persona, search_depth, loop_count
    Writes:  confidence_score, feedback, loop_count (incremented)
    """
    current_loop = state.get("loop_count", 0)
    new_loop_count = current_loop + 1

    logger.info("Evaluator started (Loop %d → %d)", current_loop, new_loop_count)

    # Safety net: router already stops retries at MAX_LOOP_COUNT, but guard here
    # if something bypasses the router.
    if new_loop_count > MAX_LOOP_COUNT:
        logger.warning(
            "Loop limit exceeded (%d). Forcing finalization.", new_loop_count
        )
        return {
            "loop_count": new_loop_count,
            "confidence_score": 0.5,
            "feedback": "Loop limit exceeded. Report will be finalized with current state.",
        }

    structured_llm = get_llm().with_structured_output(EvaluationOutput)

    source_count = len(state.get("sources", []))
    section_count = state.get("draft", "").count("\n## ") 

    messages = [
        SystemMessage(content=(
            f"You are a strict quality reviewer for research reports.\n"
            f"Evaluate the draft for a '{state['persona']}' "
            f"with research depth {state['search_depth']}/5.\n\n"
            f"Evaluation criteria:\n"
            f"1. Source coverage: Are statements backed by sources? "
            f"   (Available: {source_count} sources)\n"
            f"2. Completeness: Are all relevant aspects for a "
            f"   {state['persona']} covered?\n"
            f"3. Structure: Are {section_count} sections sufficient "
            f"   for depth {state['search_depth']}/5?\n"
            f"4. Coherence: Are key findings supported by the report content?\n\n"
            f"Be critical. A score >= 0.7 means: ready for publication."
        )),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n\n"
            f"## Draft to evaluate\n\n{state.get('draft', '')}\n\n"
            f"## Available sources ({source_count})\n"
            + "\n".join(
                f"- {s.get('title', 'N/A')} | Score: {s.get('relevance_score', 0):.2f}"
                for s in state.get("sources", [])
            )
        )),
    ]

    try:
        output: EvaluationOutput = structured_llm.invoke(messages)

        logger.info(
            "Evaluator: score=%.2f, reasoning='%s'",
            output.confidence_score,
            output.reasoning[:80],
        )

        return {
            "confidence_score": output.confidence_score,
            "feedback": _build_feedback(output),
            "loop_count": new_loop_count,
        }

    except Exception as e:
        logger.error("Evaluator node failed: %s", e)
        raise


def _build_feedback(output: EvaluationOutput) -> str:
    """Formats EvaluationOutput as a clear improvement request for the synthesizer."""
    lines = [
        f"Confidence Score: {output.confidence_score:.2f}/1.0",
        f"Assessment: {output.reasoning}",
    ]

    if output.improvement_suggestions:
        lines.append("\nConcrete improvements:")
        for suggestion in output.improvement_suggestions:
            lines.append(f"- {suggestion}")

    return "\n".join(lines)

def finalizer_node(state: ResearchState) -> dict:
    """
    Transfers the validated draft into the final report.

    Reads:   draft, confidence_score, loop_count, sources
    Writes:  final_report
    """
    metadata = (
        f"\n\n---\n"
        f"*Confidence Score: {state.get('confidence_score', 0):.2f} | "
        f"Loops: {state.get('loop_count', 0)} | "
        f"Sources: {len(state.get('sources', []))}*"
    )

    final_report = state.get("draft", "") + metadata

    logger.info(
        "Report finalized. Score=%.2f, Loops=%d, Sources=%d",
        state.get("confidence_score", 0),
        state.get("loop_count", 0),
        len(state.get("sources", [])),
    )

    return {"final_report": final_report}

from langgraph.types import interrupt


def human_review_node(state: ResearchState) -> dict:
    """
    HITL breakpoint: pauses the graph and hands queries to the UI.
    Resumes once the user has validated the queries in Streamlit.

    Reads:   search_queries
    Writes:  validated_search_queries
    """
    # interrupt() pauses here. The dict payload is visible to the UI.
    # graph.invoke(Command(resume={...}), config) resumes the graph.
    approved: dict = interrupt({
        "search_queries": state["search_queries"],
        "message": "Please validate or edit the search queries.",
    })

    validated = approved.get("validated_search_queries")
    if validated is None:
        logger.warning(
            "Resume payload missing 'validated_search_queries'; falling back to planner output."
        )
        validated = state["search_queries"]

    logger.info(
        "HITL completed: %d queries validated.", len(validated)
    )
    return {"validated_search_queries": validated}
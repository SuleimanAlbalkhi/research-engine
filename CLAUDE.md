# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`research-engine` is a LangGraph agent that turns a topic into a sourced research
report. A linear pipeline plans search queries, pauses for human approval, searches
the web in parallel, synthesizes a draft, and then self-corrects in a scored
evaluate→retry loop before finalizing. LLM and search backends are swappable
(local Ollama or OpenAI; DuckDuckGo or Tavily) via `.env`.

The project is **pre-integration**: there is no dependency manifest, no entry point
or UI (the code references a Streamlit front-end that doesn't exist yet), and no
tests. See [Current state](#current-state--known-gaps) — several cross-file symbol
names don't match yet, so the graph does not import as-is.

## Architecture

The whole workflow is a single compiled `StateGraph`. The three concepts that
require reading multiple files together:

### 1. The shared-state "whiteboard"
[core/state.py](core/state.py) defines `ResearchState`, a `TypedDict` every node
reads from and writes to. Each node returns a partial dict that LangGraph merges in.
- **Required inputs** (must be passed to `graph.invoke()`): `topic`, `persona`,
  `search_depth`, `output_format`.
- **`NotRequired` fields** are filled in by nodes as the graph runs.
- **Reducers matter:** `retrieved_docs` is `Annotated[List, operator.add]`, so the
  researcher's results are *appended*, not replaced. This is what lets evidence
  accumulate across retry loops — do not "fix" a node to overwrite it.

### 2. The graph pipeline
[core/graph.py](core/graph.py) wires the nodes from [core/nodes.py](core/nodes.py):

```
START → planner → human_review → researcher → synthesizer → evaluator → ┐
                                      ▲                                  │
                                      └──────── retry ───────────────────┤ (conditional)
                                                                         │
                                                          finalizer → END┘
```

- `planner_node` — turns topic/persona/`search_depth` into N search queries
  (`_queries_for_depth` maps depth 1–5 → 2/3/5/7/10 queries).
- `human_review_node` — **HITL breakpoint** (see below).
- `researcher_node` — runs validated queries concurrently via
  `ThreadPoolExecutor(max_workers=5)`; per-query errors are isolated and dropped,
  and it raises only if *every* query fails.
- `synthesizer_node` — builds a Markdown `draft` + `sources`; on retry it injects the
  evaluator's `feedback` into the prompt.
- `evaluator_node` — scores the draft (`confidence_score`), writes `feedback`, and
  increments `loop_count`.
- `finalizer_node` — appends a metadata footer to produce `final_report`.

### 3. The self-correction loop (two constants kept in sync across two files)
`_route_after_evaluator` in [core/graph.py](core/graph.py) routes back to
`researcher` while `confidence_score < 0.7 AND loop_count < 3`, otherwise to
`finalizer`. `evaluator_node` in [core/nodes.py](core/nodes.py) hard-stops at
`loop_count >= 3` regardless of score. **The `0.7` threshold and the `3`-loop cap
are duplicated in both files** — change them together or the router and the node
will disagree.

### Human-in-the-loop (HITL)
`human_review_node` calls LangGraph's `interrupt({...})`, which pauses the graph and
surfaces the generated `search_queries` to the caller. Execution resumes only when
the caller re-invokes with `Command(resume={"validated_search_queries": [...]})`.
Because this relies on checkpointing, **every invocation needs a `thread_id`**, and
the graph is compiled with a `MemorySaver` by default (`build_graph` accepts a
production checkpointer like `SqliteSaver`/`PostgresSaver`).

### Providers & configuration
- [core/config.py](core/config.py) — `Settings` (pydantic-settings) loads `.env`.
  `get_settings()` is an `lru_cache` singleton; a `model_validator` enforces that the
  required API key is present for the chosen provider. In tests call
  `get_settings.cache_clear()` after changing env.
- [core/providers.py](core/providers.py) — `get_llm()` / `get_search_tool()` are
  `lru_cache` factories returning the abstract `BaseChatModel` / `BaseTool`. **Nodes
  depend only on these factories and the abstract return types, never on a concrete
  provider class.** Provider packages are imported lazily *inside* the factories so
  you only need the deps for the backend you actually use.

### Structured outputs
[schemas/output_schemas.py](schemas/output_schemas.py) holds the Pydantic models that
nodes request via `llm.with_structured_output(...)`. Each LLM-calling node has a
matching schema (planner→queries, synthesizer→sections/findings/sources,
evaluator→score/feedback). This is the contract between a node and its schema — they
must stay name- and field-aligned (they currently don't; see below).

## Running it

There is no manifest yet. Run from the **repository root** so the absolute `core.*`
and `schemas.*` imports resolve (there are no `__init__.py` files; this works as a
namespace package only from the root).

```bash
# 1. Dependencies (inferred from imports — no requirements.txt exists yet):
pip install langgraph langchain-core langchain-community langchain-ollama \
            langchain-openai langchain-tavily duckduckgo-search \
            pydantic pydantic-settings

# 2. Config
cp .env.example .env          # defaults: Ollama (deepseek-r1:8b) + DuckDuckGo

# 3. If using the default Ollama provider, have it running with the model pulled:
ollama serve
ollama pull deepseek-r1:8b
```

Invoke the graph programmatically (the HITL pause makes this a two-call sequence):

```python
from langgraph.types import Command
from core.graph import graph

config = {"configurable": {"thread_id": "demo-1"}}
state = graph.invoke(
    {"topic": "...", "persona": "...", "search_depth": 3, "output_format": "..."},
    config,
)
# graph pauses in human_review; inspect the interrupt payload, then resume:
result = graph.invoke(
    Command(resume={"validated_search_queries": [...]}),
    config,
)
print(result["final_report"])
```

## Conventions

- **Provider-agnostic nodes:** add a new LLM/search backend by extending the factory
  in [core/providers.py](core/providers.py) and the `Literal` + validator in
  [core/config.py](core/config.py) — never import a concrete provider in a node.
- **State changes are contracts:** adding a field means updating `ResearchState`, the
  producing node, and every consuming node. Decide whether it needs a reducer.
- **Structured output over parsing:** get data out of the LLM with a Pydantic schema
  and `with_structured_output`, not by string-parsing responses.
- **Logging, not printing:** modules use `logging.getLogger(__name__)`.

## Current state / known gaps

The code is mid-refactor and the node↔schema and node↔state contracts are out of
sync. Before the graph can import/run, these need reconciling (names as currently
written):

- [core/nodes.py](core/nodes.py) imports `SearchQueriesOutput` and uses
  `SynthesisOutput` / `EvaluationOutput`, but
  [schemas/output_schemas.py](schemas/output_schemas.py) defines
  `SearcherQueriesOutput` / `SynthesizerOutput` / `EvaluatorOutput` (and only the
  first is imported).
- State key mismatch: [core/state.py](core/state.py) declares `confidence_source`,
  while nodes and the router read/write `confidence_score`.
- In [schemas/output_schemas.py](schemas/output_schemas.py): `SourceDocument`'s
  `url`/`title`/`snippet` are missing `str` type annotations; `ReportSection` defines
  `sources_urls` but the synthesizer reads `section.source_urls`; `key_findings` is
  typed `str` but treated as a list; and `validate_retry_logic` is defined at module
  scope instead of inside `EvaluatorOutput`.

When you touch these areas, treat the above as the intended-but-incomplete wiring
rather than working behavior to preserve.

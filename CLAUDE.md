# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`research-engine` is a LangGraph agent that turns a topic into a sourced research report. A linear pipeline plans search queries, pauses for human approval, searches the web in parallel, synthesizes a draft, then self-corrects in a scored evaluate→retry loop before finalizing. LLM and search backends are swappable (local Ollama or OpenAI; DuckDuckGo or Tavily) via `.env`.

## Running it

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env   # defaults: Ollama (qwen3:8b) + DuckDuckGo

# 3. If using Ollama (default), start the server and pull the model
ollama serve
ollama pull qwen3:8b

# 4. Run
python run.py
```

`run.py` invokes the graph, prints the generated queries, waits for terminal input to approve or replace them, then prints the final report. It is the canonical manual test harness.

## Architecture

The entire workflow is a single compiled `StateGraph`. The three cross-file concepts:

### 1. Shared state (`core/state.py`)

`ResearchState` is a `TypedDict` every node reads from and writes to. Each node returns a partial dict that LangGraph merges in.

**Required inputs** (must be passed to `graph.invoke()`): `topic`, `persona`, `search_depth`, `output_format`.

**Reducer:** `retrieved_docs` is `Annotated[List, operator.add]` — the researcher's results are *appended*, not replaced, so evidence accumulates across retry loops. Do not change this to an overwrite.

### 2. The graph pipeline (`core/graph.py`, `core/nodes.py`)

```
START → planner → human_review → researcher → synthesizer → evaluator → ┐
                                      ▲                                  │
                                      └──────── retry ───────────────────┤ (conditional)
                                                                         │
                                                          finalizer → END┘
```

| Node | Reads | Writes |
|---|---|---|
| `planner_node` | `topic`, `persona`, `search_depth` | `search_queries` |
| `human_review_node` | `search_queries` | `validated_search_queries` |
| `researcher_node` | `validated_search_queries` | `retrieved_docs` (appended) |
| `synthesizer_node` | `retrieved_docs`, `topic`, `persona`, `output_format`, `feedback` (optional) | `draft`, `sources` |
| `evaluator_node` | `draft`, `sources`, `topic`, `persona`, `search_depth`, `loop_count` | `confidence_score`, `feedback`, `loop_count` |
| `finalizer_node` | `draft`, `confidence_score`, `loop_count`, `sources` | `final_report` |

`_queries_for_depth` in `nodes.py` maps `search_depth` 1–5 → 2/3/5/7/10 queries. The researcher runs all queries concurrently with `ThreadPoolExecutor(max_workers=5)`; individual query failures are isolated and dropped; it only raises if *every* query fails.

On retry, `synthesizer_node` detects `feedback` in state and injects the evaluator's critique into the prompt so the new draft explicitly addresses the previous shortcomings.

### 3. Self-correction loop constants (`core/constants.py`)

```python
MIN_CONFIDENCE_SCORE: float = 0.7
MAX_LOOP_COUNT: int = 3
```

`_route_after_evaluator` in `graph.py` and `evaluator_node` in `nodes.py` both import these from `core/constants.py` — this is the **single source of truth**. Change the values only here.

### Human-in-the-loop (HITL)

`human_review_node` calls `interrupt({...})`, which pauses the graph and surfaces `search_queries` to the caller. Execution resumes only when the caller re-invokes with:

```python
graph.invoke(Command(resume={"validated_search_queries": [...]}), config)
```

Every invocation requires a `thread_id` in the config because HITL depends on checkpointing. The graph is compiled with `MemorySaver` by default; `build_graph(checkpointer=...)` accepts `SqliteSaver`/`PostgresSaver` for production.

### Providers & configuration

- `core/config.py` — `Settings` (pydantic-settings) loads `.env`. `get_settings()` is an `lru_cache` singleton; a `model_validator` enforces that the required API key is present for the chosen provider. Call `get_settings.cache_clear()` after patching env in tests.
- `core/providers.py` — `get_llm()` / `get_search_tool()` are `lru_cache` factories returning `BaseChatModel` / `BaseTool`. Provider packages are imported lazily inside the factories. **Nodes depend only on these factories and the abstract return types — never import a concrete provider class in a node.**
- Optional LangSmith tracing: set `LANGCHAIN_TRACING_V2=true` and `LANGSMITH_API_KEY` in `.env`.

### Structured outputs (`schemas/output_schemas.py`)

Each LLM-calling node uses `llm.with_structured_output(Schema)`:

| Node | Schema |
|---|---|
| `planner_node` | `SearchQueriesOutput` |
| `synthesizer_node` | `SynthesizerOutput` (imported as `SynthesisOutput` in nodes) |
| `evaluator_node` | `EvaluatorOutput` (imported as `EvaluationOutput` in nodes) |

`SynthesizerOutput` depends on `ReportSection` (uses `source_urls`) and `SourceDocument`. The schema is the contract between a node and the LLM — field names must match what the node reads off the output object.

## Conventions

- **State changes are contracts:** adding a field means updating `ResearchState`, the producing node, every consuming node, and deciding whether it needs a reducer.
- **Structured output over parsing:** extract LLM data with a Pydantic schema and `with_structured_output`, never by string-parsing.
- **Logging, not printing:** all modules use `logging.getLogger(__name__)`. Entry-point scripts must call `logging.basicConfig(...)` or the pipeline runs silently.
- **Provider-agnostic nodes:** add a new backend by extending the factory in `core/providers.py` and the `Literal` + validator in `core/config.py`.

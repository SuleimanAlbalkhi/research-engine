# Research Engine

A LangGraph agent that transforms a topic into a sourced research report — with human query approval, parallel web search, and a scored self-correction loop.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-latest-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

![Demo](docs/demo.png)
> *Add screenshot after first successful run*

---

## Architecture

```
START → planner → human_review → researcher → synthesizer → evaluator → ┐
                                      ▲                                  │
                                      └──────── retry ───────────────────┤ (conditional)
                                                                         │
                                                          finalizer → END┘
```

Two design decisions worth noting. `retrieved_docs` in the shared state uses an `Annotated[List, operator.add]` reducer rather than a plain overwrite — this means every retry iteration appends new search results on top of previous ones, so the synthesizer has progressively more evidence to work with. `MIN_CONFIDENCE_SCORE` and `MAX_LOOP_COUNT` live in `core/constants.py` as the single source of truth; both the router in `graph.py` and the guard in `evaluator_node` import from there, so there is no risk of the two diverging.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph `StateGraph` |
| LLM backends | Ollama (local) · OpenAI |
| Search backends | DuckDuckGo · Tavily |
| Structured outputs | Pydantic v2 + `with_structured_output` |
| Checkpointing (HITL) | LangGraph `MemorySaver` / `SqliteSaver` / `PostgresSaver` |
| Config | pydantic-settings (`.env`) |

---

## Project Structure

```
research-engine/
├── run.py                      # CLI entry point — runs the full pipeline interactively
├── requirements.txt            # Pinned dependencies
├── .env.example                # All supported config variables with defaults
│
├── core/
│   ├── state.py                # ResearchState TypedDict — the shared whiteboard
│   ├── graph.py                # StateGraph wiring and conditional routing
│   ├── nodes.py                # All six node functions
│   ├── providers.py            # LLM and search tool factories (lru_cache, abstract types)
│   ├── config.py               # pydantic-settings Settings class, get_settings() singleton
│   └── constants.py            # MIN_CONFIDENCE_SCORE and MAX_LOOP_COUNT
│
└── schemas/
    └── output_schemas.py       # Pydantic models for structured LLM outputs
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt
# Only install the provider packages you actually need —
# langchain-ollama if using Ollama, langchain-openai if using OpenAI,
# langchain-tavily if using Tavily (DuckDuckGo needs no key).

# 2. Configure
cp .env.example .env

# 3. If using Ollama (default), start the server and pull the model
ollama serve
ollama pull qwen3:8b

# 4. Run
python run.py
```

Run from the repository root so that `core.*` and `schemas.*` imports resolve correctly.

---

## Human-in-the-Loop

After the planner generates search queries, the graph calls LangGraph's `interrupt()`, which pauses execution and surfaces the queries to the caller. In `run.py` this is a terminal prompt; in a UI it would be a review screen. Execution resumes only when the caller re-invokes the graph with `Command(resume={"validated_search_queries": [...]})`. Because the pause is persisted via a checkpointer, every invocation must include a `thread_id` in the config — without it LangGraph cannot match the resumed invocation to the correct saved state.

---

## Self-Correction Loop

After synthesis, the evaluator scores the draft from 0.0 to 1.0 and writes structured improvement suggestions back to state. The router then sends the graph back to the researcher if `confidence_score < 0.7` and `loop_count < 3`; otherwise it routes to the finalizer. There are two exit conditions: the draft reaches acceptable quality, or the loop cap is hit as a hard safety net. On each retry the synthesizer receives the previous evaluator feedback and is explicitly asked to address it.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | LLM backend: `ollama` or `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `deepseek-r1:8b` | Model name to pull and serve |
| `OLLAMA_NUM_GPU` | `0` | Layers offloaded to GPU (0 = CPU only) |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `SEARCH_PROVIDER` | `duckduckgo` | Search backend: `duckduckgo` or `tavily` |
| `TAVILY_API_KEY` | — | Required when `SEARCH_PROVIDER=tavily` |
| `LANGSMITH_API_KEY` | — | Optional — enables LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | `false` | Set `true` to activate tracing |
| `LANGCHAIN_PROJECT` | `research-engine` | LangSmith project name |

---

## What I'd do differently at scale

- Enable LangSmith tracing in production — the config already supports it (`LANGCHAIN_TRACING_V2`, `LANGSMITH_API_KEY`); it's the fastest way to debug prompt regressions and score drift across runs.
- Replace `MemorySaver` with `PostgresSaver` so graph state survives restarts and multiple workers can share checkpoints without contention.
- Rewrite the researcher node with `asyncio` and an async search client — the current `ThreadPoolExecutor` works but carries thread overhead that async I/O avoids entirely.
- Expose `MIN_CONFIDENCE_SCORE` and `MAX_LOOP_COUNT` as runtime config rather than compile-time constants, so quality thresholds can be tuned per-use-case without a code change.

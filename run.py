import logging
from langgraph.types import Command
from core.graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

graph = build_graph()
config = {"configurable": {"thread_id": "run-1"}}

state = graph.invoke(
    {
        "topic": "LLM agents in production",
        "persona": "software engineer",
        "search_depth": 1,
        "output_format": "markdown report",
    },
    config,
)

print("\n=== Generated search queries ===")
for i, q in enumerate(state["search_queries"], 1):
    print(f"  {i}. {q}")

raw = input("\nPress Enter to approve, or type comma-separated replacements: ").strip()
validated_queries = (
    [q.strip() for q in raw.split(",") if q.strip()]
    if raw else state["search_queries"]
)

result = graph.invoke(
    Command(resume={"validated_search_queries": validated_queries}),
    config,
)
print("\n=== Final Report ===")
print(result["final_report"])

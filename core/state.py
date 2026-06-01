from __future__ import annotations

import operator 
from typing import Annotated, List
from typing_extensions import NotRequired, TypedDict

class ResearchState(TypedDict):
    """
    The shared whiteboard for the entire research workflow. 
    Fields without the `NotRequired` attribute must be passed to `graph.invoke()`.
    Fields marked as `NotRequired` are populated by nodes during execution. Reducers (`Annotated` and `operator.add`) accumulate lists over loop iterations.
    """

    # Inputs
    topic: str
    persona: str
    search_depth: int
    output_format: str

    # Planner Node
    search_queries: NotRequired[List[str]]
    validated_search_queries: NotRequired[List[str]]

    # Researcher Node
    # Reducer: `operator.add` appends new documents instead of replacing them.
    # Important for the self-correction loop – documents accumulate.
    retrieved_docs: NotRequired[Annotated[List[dict], operator.add]]

    # Synthesizer Node
    draft: NotRequired[str]
    sources: NotRequired[List[dict]] # JSON source directory from retrieved_docs 

    # Evaluater Node
    confidence_score: NotRequired[float]
    feedback: NotRequired[str]
    loop_count: NotRequired[int]

    # Final Output
    final_report: NotRequired[str]
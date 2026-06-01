from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# Atomic building blocks
class SourceDocument(BaseModel):
    """Represents a single researched source."""
    url: str = Field(description="The URL of the source document.")
    title: str = Field(description="The title of the source document.")
    snippet: str = Field(description="A brief snippet or summary of the source document.")
    relevance_score: float = Field( ge = 0.0, le = 1.0, description="A relevance score between 0 and 1 indicating how relevant the source is to the research topic.")

class ReportSection(BaseModel):
    """Represents a section of the final report."""
    title: str = Field(description="The title of the report section.")
    content: str = Field(description="The content of the report section, which may include synthesized information from multiple sources.")
    source_urls: list[str] = Field(default_factory=list, description="A list of URLs for the source documents that were used to create this section.")

# Node-Outputs

class SearchQueriesOutput(BaseModel):
    """Output from the Planner node."""
    queries: list[str] = Field(min_length=1, description="Generated search queries derived from topic and search_depth")
    reasoning: str = Field(description="The reasoning behind the generated search queries.")

    @field_validator('queries')
    @classmethod
    def queries_not_empty_strings(cls, v: list[str]) -> list[str]:
        if any(q.strip() == "" for q in v):
            raise ValueError("All queries must be non-empty strings.")
        return v
    
class SynthesizerOutput(BaseModel):
    """Output from the Synthesizer node."""
    sections: list[ReportSection] = Field(min_length=1, description="A list of report sections that together form the retrieved_docs.")
    key_findings: list[str] = Field(min_length=3, max_length=5, description="3–5 key takeaways from the report sections.")
    sources: list[SourceDocument] = Field(default_factory=list, description="A list of source documents that were synthesized into the report sections.")

class EvaluatorOutput(BaseModel):
    """Output from the Evaluator node."""
    confidence_score: float = Field(ge=0.0, le=1.0, description="A confidence score between 0 and 1 indicating the evaluator's confidence in the quality of the draft report.")
    reasoning: str = Field(description="Constructive feedback on how to improve the draft report.")
    improvement_suggestions: list[str] = Field(default_factory=list, description="A list of specific suggestions for improving the draft report.")

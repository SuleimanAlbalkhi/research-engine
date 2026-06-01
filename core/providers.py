from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from core.config import get_settings


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Returns the configured LLM.
    Nodes import only this return type – never the concrete class."""
    settings = get_settings()

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
            #num_gpu=settings.ollama_num_gpu,
        )

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )

    raise ValueError(f"Unknown LLM-Provider: {settings.llm_provider!r}")


@lru_cache(maxsize=1)
def get_search_tool() -> BaseTool:
    """Returns the configured Search-Tool."""
    settings = get_settings()

    if settings.search_provider == "duckduckgo":
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun()

    if settings.search_provider == "tavily":
        from langchain_tavily import TavilySearchResults
        return TavilySearchResults(
            api_key=settings.tavily_api_key,
            max_results=5,
        )

    raise ValueError(f"Unknown Search-Provider: {settings.search_provider!r}")
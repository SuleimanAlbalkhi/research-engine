from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    llm_provider: Literal["ollama", "openai"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_num_gpu: int = 0  # 0 = CPU only; set in .env for partial GPU offload
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # Search
    search_provider: Literal["duckduckgo", "tavily"] = "duckduckgo"
    tavily_api_key: str | None = None

    # LangSmith Tracing (optional)
    langsmith_api_key: str | None = None
    langchain_tracing_v2: bool = False
    langchain_project: str = "research-engine"

    @model_validator(mode="after")
    def validate_api_keys(self) -> Settings:
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "llm_provider='openai' requires OPENAI_API_KEY in .env"
            )
        if self.search_provider == "tavily" and not self.tavily_api_key:
            raise ValueError(
                "search_provider='tavily' requires TAVILY_API_KEY in .env"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton – call get_settings.cache_clear() in tests after env changes."""
    return Settings()

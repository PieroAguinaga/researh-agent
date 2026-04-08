"""
config/settings.py

Centralised, type-safe configuration powered by Pydantic Settings.
All values are read from environment variables or the .env file.

Usage:
    from config.settings import settings
    print(settings.azure_chat_deployment)
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Azure OpenAI ──────────────────────────────────────────────────────────
    azure_chat_deployment: str = Field(..., description="Chat model deployment name")
    azure_openai_endpoint: str = Field(..., description="Azure OpenAI resource endpoint")
    azure_openai_api_key: str = Field(..., description="Azure OpenAI API key")
    azure_openai_api_version: str = Field(default="2024-02-15-preview")
    azure_embedding_deployment: str = Field(default="text-embedding-ada-002")
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=2048)

    # ── Supabase ──────────────────────────────────────────────────────────────
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_service_role_key: str = Field(..., description="Supabase service role key")

    # Table names — override in .env if you use a different naming convention
    supabase_conversations_table: str = Field(default="conversation_history")
    supabase_papers_table: str = Field(default="papers")
    supabase_embeddings_table: str = Field(default="paper_embeddings")
    supabase_embeddings_query: str = Field(default="match_paper_embeddings")

    # ── Paper APIs ────────────────────────────────────────────────────────────
    arxiv_max_results: int = Field(default=8)

    # ── Flask ─────────────────────────────────────────────────────────────────
    flask_secret_key: str = Field(default="change-me-in-production")
    flask_debug: bool = Field(default=False)
    flask_host: str = Field(default="0.0.0.0")
    flask_port: int = Field(default=5000)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton. Avoids re-parsing .env on every call."""
    return Settings()


# Module-level alias 
settings = get_settings()

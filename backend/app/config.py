"""Load settings from the environment / .env.local.

Secrets stay out of the repo. The frontend already talks to port 8787, so that
is the default here.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env.local", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8787
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_api_token: str = ""
    chat_model: str = "qwen3.8-27b-uncensored-mlx"
    embed_model: str = "text-embedding-nomic-embed-text-v1.5"

    brave_api_key: str = ""

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    max_files_per_upload: int = 10
    max_file_bytes: int = 10_000_000
    chunk_size: int = 800
    chunk_overlap: int = 80
    retrieve_top_k: int = 8
    max_agent_steps: int = 5
    embed_workers: int = 3

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

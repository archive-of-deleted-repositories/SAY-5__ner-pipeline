"""Application settings loaded from env vars."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NER_",
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql+psycopg://ner:ner@localhost:5432/ner",
        description="SQLAlchemy URL for the application database.",
    )

    # Model
    model_name: str = Field(
        default="dslim/distilbert-NER",
        description="HuggingFace model id for the NER pipeline.",
    )
    model_revision: str | None = Field(
        default=None,
        description="Optional model revision pin (commit sha or tag).",
    )
    model_device: str = Field(
        default="cpu",
        description="Torch device for inference (cpu / cuda / mps).",
    )

    # Service
    log_level: str = Field(default="INFO")
    request_id_header: str = Field(default="x-request-id")

    # Eval
    eval_dataset: str = Field(
        default="eriktks/conll2003",
        description="HuggingFace dataset id for eval (CoNLL-2003).",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()

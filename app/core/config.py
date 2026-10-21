from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_tokens: Annotated[list[str], NoDecode] = Field(
        description="Comma-separated App Tokens that may be exchanged for JWTs"
    )
    jwt_secret: SecretStr
    jwt_algorithm: str = "HS256"
    access_token_ttl_seconds: int = Field(default=15 * 60, gt=0)
    refresh_token_ttl_seconds: int = Field(default=24 * 60 * 60, gt=0)

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("password")
    neo4j_database: str = "neo4j"

    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    vision_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, gt=0)
    llm_concurrency: int = Field(default=4, gt=0)
    llm_timeout_seconds: float = Field(default=120, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)

    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    chunk_size_chars: int = Field(default=12_000, gt=0)
    chunk_overlap_chars: int = Field(default=800, ge=0)

    pdf_max_visual_items: int = Field(default=30, ge=0)
    pdf_min_image_side_px: int = Field(default=100, gt=0)
    pdf_render_dpi: int = Field(default=150, gt=0)
    pdf_scanned_page_max_chars: int = Field(default=50, ge=0)
    pdf_drawings_render_threshold: int = Field(default=30, gt=0)

    character_candidate_limit: int = Field(default=10, gt=0)
    # On Neo4j's cosine score scale: score = (1 + cos) / 2.
    relation_type_candidate_score: float = Field(default=0.75, ge=0, le=1)

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _empty_key_as_unset(cls, value: object) -> object:
        # docker compose passes `${VAR:-}` through as an empty string.
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def _empty_base_url_as_default(cls, value: object) -> object:
        # Always explicit: given None, the OpenAI SDK would itself read the (possibly empty) OPENAI_BASE_URL.
        if isinstance(value, str) and not value.strip():
            return cls.model_fields["openai_base_url"].default
        return value

    @field_validator("app_tokens", mode="before")
    @classmethod
    def _split_app_tokens(cls, value: object) -> object:
        if isinstance(value, str):
            return [token.strip() for token in value.split(",") if token.strip()]
        return value

    @field_validator("app_tokens")
    @classmethod
    def _require_app_tokens(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("at least one app token must be configured")
        return value

    @field_validator("jwt_secret")
    @classmethod
    def _require_strong_secret(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 32:
            raise ValueError("JWT secret must be at least 32 characters long")
        return value

import pytest
from pydantic import ValidationError

from app.core.config import Settings

SECRET = "x" * 32


def make(**overrides: object) -> Settings:
    return Settings(_env_file=None, **{"app_tokens": "a", "jwt_secret": SECRET, **overrides})


def test_app_tokens_are_parsed_from_comma_separated_string() -> None:
    assert make(app_tokens=" one, two ,,").app_tokens == ["one", "two"]


def test_app_tokens_are_required() -> None:
    with pytest.raises(ValidationError):
        make(app_tokens=" , ")


def test_short_jwt_secret_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make(jwt_secret="short")


def test_empty_openai_values_from_compose_are_treated_as_unset() -> None:
    settings = make(openai_api_key="", openai_base_url="  ")

    assert settings.openai_api_key is None
    assert settings.openai_base_url == "https://api.openai.com/v1"


def test_environment_variables_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_TOKENS", "env-token")
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "256")

    settings = Settings(_env_file=None)

    assert settings.app_tokens == ["env-token"]
    assert settings.embedding_dimensions == 256

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
    monkeypatch.setenv("JUDGE_MODEL", "judge-x")
    monkeypatch.setenv("CHARACTER_CANDIDATE_LIMIT", "3")
    monkeypatch.setenv("RELATION_TYPE_CANDIDATE_SCORE", "0.8")

    settings = Settings(_env_file=None)

    assert settings.app_tokens == ["env-token"]
    assert settings.embedding_dimensions == 256
    assert settings.judge_model == "judge-x"
    assert settings.character_candidate_limit == 3
    assert settings.relation_type_candidate_score == 0.8


def test_dedup_defaults() -> None:
    settings = make()

    assert settings.judge_model == "gpt-4o-mini"
    assert settings.character_candidate_limit == 10
    assert settings.relation_type_candidate_score == 0.75


REMOVED = ["CHARACTER_" + "MATCH_SCORE", "CHARACTER_" + "AUTO_MERGE_SCORE", "RELATION_TYPE_" + "MATCH_SCORE"]


@pytest.mark.parametrize("name", REMOVED)
def test_old_score_settings_are_ignored(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "0.5")

    settings = make()

    assert not hasattr(settings, name.lower())


def test_model_defaults() -> None:
    settings = make()

    assert settings.llm_model == "gpt-4.1-mini"
    assert settings.vision_model == "gpt-4o-mini"
    assert settings.judge_model == "gpt-4o-mini"


def test_limit_defaults() -> None:
    settings = make()

    assert settings.max_pdf_pages == 100
    assert settings.max_chunks == 50
    assert settings.pdf_max_render_side_px == 2048
    assert settings.stale_processing_seconds == 1800


@pytest.mark.parametrize("algorithm", ["none", "RS256"])
def test_jwt_algorithm_is_restricted_to_hmac(algorithm: str) -> None:
    with pytest.raises(ValidationError):
        make(jwt_algorithm=algorithm)


def test_hmac_jwt_algorithm_is_accepted() -> None:
    assert make(jwt_algorithm="HS512").jwt_algorithm == "HS512"


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        make(chunk_size_chars=100, chunk_overlap_chars=100)

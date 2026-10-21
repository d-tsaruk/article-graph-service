from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.llm.base import LLMError
from app.llm.openai_client import OpenAILLMClient, build_openai_client
from app.llm.schemas import ExtractionResult, ImageAnalysis
from tests.factories import harry_and_ron


class StubCompletions:
    def __init__(self, message: SimpleNamespace) -> None:
        self.message = message
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.message)])


class StubEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        # Returned out of order on purpose: the API contract only guarantees `index`.
        data = [SimpleNamespace(index=i, embedding=[float(i)]) for i in range(len(kwargs["input"]))]
        return SimpleNamespace(data=list(reversed(data)))


def make_client(message: SimpleNamespace) -> tuple[OpenAILLMClient, StubCompletions, StubEmbeddings]:
    completions = StubCompletions(message)
    embeddings = StubEmbeddings()
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions), embeddings=embeddings)
    client = OpenAILLMClient(
        sdk,  # type: ignore[arg-type]
        llm_model="extract-model",
        vision_model="vision-model",
        embedding_model="embed-model",
        embedding_dimensions=8,
        concurrency=2,
    )
    return client, completions, embeddings


async def test_extract_uses_structured_output_and_known_types() -> None:
    expected = harry_and_ron()
    client, completions, _ = make_client(SimpleNamespace(parsed=expected, refusal=None))

    result = await client.extract("Harry met Ron.", known_relation_types=["FRIEND_OF", "ENEMY_OF"])

    assert result == expected
    call = completions.calls[0]
    assert call["model"] == "extract-model"
    assert call["response_format"] is ExtractionResult
    assert call["temperature"] == 0
    user_message = call["messages"][-1]["content"]
    assert "Harry met Ron." in user_message
    assert "FRIEND_OF, ENEMY_OF" in user_message


async def test_extract_raises_on_refusal() -> None:
    client, _, _ = make_client(SimpleNamespace(parsed=None, refusal="I can't help with that"))

    with pytest.raises(LLMError, match="refused"):
        await client.extract("text", known_relation_types=[])


async def test_analyze_image_sends_base64_data_url_to_vision_model() -> None:
    analysis = ImageAnalysis(is_relevant=True, description="A map")
    client, completions, _ = make_client(SimpleNamespace(parsed=analysis, refusal=None))

    result = await client.analyze_image(b"\x89PNG", "image/png", context="page 2")

    assert result == analysis
    call = completions.calls[0]
    assert call["model"] == "vision-model"
    assert call["response_format"] is ImageAnalysis
    content = call["messages"][-1]["content"]
    image_part = next(part for part in content if part["type"] == "image_url")
    assert image_part["image_url"]["url"] == "data:image/png;base64,iVBORw=="


async def test_embed_preserves_input_order() -> None:
    client, _, embeddings = make_client(SimpleNamespace(parsed=None, refusal=None))

    vectors = await client.embed(["a", "b", "c"])

    assert vectors == [[0.0], [1.0], [2.0]]
    assert embeddings.calls[0]["model"] == "embed-model"
    assert embeddings.calls[0]["dimensions"] == 8


async def test_embed_empty_input_makes_no_call() -> None:
    client, _, embeddings = make_client(SimpleNamespace(parsed=None, refusal=None))

    assert await client.embed([]) == []
    assert embeddings.calls == []


def test_build_client_ignores_empty_base_url_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # docker compose passes `OPENAI_BASE_URL=` through; the SDK would otherwise read it and get an empty URL.
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    settings = Settings(_env_file=None, app_tokens="a", jwt_secret="x" * 32, openai_api_key="sk-test")

    client = build_openai_client(settings)

    assert str(client._sdk.base_url) == "https://api.openai.com/v1/"


def test_build_client_uses_configured_base_url() -> None:
    settings = Settings(
        _env_file=None,
        app_tokens="a",
        jwt_secret="x" * 32,
        openai_api_key="sk-test",
        openai_base_url="http://localhost:11434/v1",
    )

    assert str(build_openai_client(settings)._sdk.base_url) == "http://localhost:11434/v1/"

import logging
from types import SimpleNamespace
from typing import Any

import openai
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.llm import prompts
from app.llm.base import LLMError
from app.llm.judging import CharacterCard, RelationTypeCard
from app.llm.openai_client import OpenAILLMClient, build_openai_client
from app.llm.schemas import ExtractionResult, ImageAnalysis, Judgement, Verdict
from tests.factories import harry_and_ron


class StubCompletions:
    def __init__(self, message: SimpleNamespace, usage: SimpleNamespace | None = None) -> None:
        self.message = message
        self.usage = usage
        self.error: Exception | None = None
        self.choices: list[SimpleNamespace] | None = None
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        choices = [SimpleNamespace(message=self.message)] if self.choices is None else self.choices
        return SimpleNamespace(choices=choices, usage=self.usage)


class StubEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        # Returned out of order on purpose: the API contract only guarantees `index`.
        data = [SimpleNamespace(index=i, embedding=[float(i)]) for i in range(len(kwargs["input"]))]
        return SimpleNamespace(data=list(reversed(data)))


def make_client(
    message: SimpleNamespace, usage: SimpleNamespace | None = None
) -> tuple[OpenAILLMClient, StubCompletions, StubEmbeddings]:
    completions = StubCompletions(message, usage)
    embeddings = StubEmbeddings()
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions), embeddings=embeddings)
    client = OpenAILLMClient(
        sdk,  # type: ignore[arg-type]
        llm_model="extract-model",
        vision_model="vision-model",
        judge_model="judge-model",
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


JUDGEMENT = Judgement(verdicts=[Verdict(item="N1", reason="same person", decision="stored", match="S1")])


async def test_judge_characters_uses_structured_output() -> None:
    client, completions, _ = make_client(SimpleNamespace(parsed=JUDGEMENT, refusal=None))
    new = [CharacterCard("N1", "Ronald", (), "Ron's full name", ("S1",))]
    known = [CharacterCard("S1", "Ron Weasley", ("Ron",), "Harry's friend")]

    result = await client.judge_characters(new, known)

    assert result == JUDGEMENT
    call = completions.calls[0]
    assert call["model"] == "judge-model"
    assert call["response_format"] is Judgement
    system, user = call["messages"]
    assert system["content"] == prompts.CHARACTER_JUDGE_SYSTEM_PROMPT.strip()
    assert user["content"] == (
        "Known characters:\n"
        "S1 | Ron Weasley | aliases: Ron | Harry's friend\n"
        "\n"
        "New characters to resolve:\n"
        "N1 | Ronald | aliases: none | Ron's full name | candidates: S1"
    )


async def test_judge_relation_types_uses_structured_output() -> None:
    client, completions, _ = make_client(SimpleNamespace(parsed=JUDGEMENT, refusal=None))
    new = [RelationTypeCard("N1", "FRIENDS_WITH", ("S1",))]
    known = [RelationTypeCard("S1", "FRIEND_OF")]

    result = await client.judge_relation_types(new, known)

    assert result == JUDGEMENT
    call = completions.calls[0]
    assert call["model"] == "judge-model"
    assert call["response_format"] is Judgement
    system, user = call["messages"]
    assert system["content"] == prompts.RELATION_TYPE_JUDGE_SYSTEM_PROMPT.strip()
    assert user["content"] == (
        "Known relation types:\nS1 | FRIEND_OF\n\nNew relation types to resolve:\nN1 | FRIENDS_WITH | candidates: S1"
    )


async def test_judge_rendering_with_no_known_items() -> None:
    client, completions, _ = make_client(SimpleNamespace(parsed=JUDGEMENT, refusal=None))

    await client.judge_relation_types([RelationTypeCard("N1", "A_OF"), RelationTypeCard("N2", "B_OF")], [])

    assert completions.calls[0]["messages"][-1]["content"] == (
        "Known relation types:\n"
        "none\n"
        "\n"
        "New relation types to resolve:\n"
        "N1 | A_OF | candidates: none\n"
        "N2 | B_OF | candidates: none"
    )


async def test_judge_raises_on_refusal() -> None:
    client, _, _ = make_client(SimpleNamespace(parsed=None, refusal="no"))

    with pytest.raises(LLMError, match="refused"):
        await client.judge_characters([CharacterCard("N1", "A", (), "")], [])


async def test_parse_logs_token_usage(caplog: pytest.LogCaptureFixture) -> None:
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3)
    client, _, _ = make_client(SimpleNamespace(parsed=JUDGEMENT, refusal=None), usage)

    with caplog.at_level(logging.INFO, logger="app.llm.openai_client"):
        await client.judge_relation_types([RelationTypeCard("N1", "A_OF")], [])

    assert "judge-model used 12 prompt + 3 completion tokens" in caplog.text


def truncated() -> openai.LengthFinishReasonError:
    return openai.LengthFinishReasonError(completion=SimpleNamespace(usage=None))  # type: ignore[arg-type]


async def test_truncated_extraction_suggests_smaller_chunks() -> None:
    client, completions, _ = make_client(SimpleNamespace(parsed=None, refusal=None))
    completions.error = truncated()

    with pytest.raises(LLMError, match="CHUNK_SIZE_CHARS"):
        await client.extract("text", known_relation_types=[])


async def test_truncated_judge_answer_gives_a_neutral_message() -> None:
    client, completions, _ = make_client(SimpleNamespace(parsed=None, refusal=None))
    completions.error = truncated()

    with pytest.raises(LLMError, match="truncated") as info:
        await client.judge_characters([CharacterCard("N1", "A", (), "")], [])

    assert "CHUNK_SIZE_CHARS" not in str(info.value)


async def test_malformed_model_output_becomes_llm_error() -> None:
    client, completions, _ = make_client(SimpleNamespace(parsed=None, refusal=None))
    try:
        Judgement.model_validate({})
    except ValidationError as exc:
        completions.error = exc

    with pytest.raises(LLMError, match="malformed"):
        await client.judge_relation_types([RelationTypeCard("N1", "A_OF")], [])


async def test_empty_choices_become_llm_error() -> None:
    client, completions, _ = make_client(SimpleNamespace(parsed=None, refusal=None))
    completions.choices = []

    with pytest.raises(LLMError, match="no choices"):
        await client.judge_relation_types([RelationTypeCard("N1", "A_OF")], [])


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

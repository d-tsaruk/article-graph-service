from typing import Protocol

from app.llm.judging import CharacterCard, RelationTypeCard
from app.llm.schemas import ExtractionResult, ImageAnalysis, Judgement


class LLMError(Exception):
    """The model backend failed or returned an unusable response."""


class LLMNotConfiguredError(LLMError):
    pass


class LLMClient(Protocol):
    async def extract(self, text: str, known_relation_types: list[str]) -> ExtractionResult: ...

    async def analyze_image(self, image: bytes, mime_type: str, context: str) -> ImageAnalysis: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def judge_characters(self, new: list[CharacterCard], known: list[CharacterCard]) -> Judgement: ...

    async def judge_relation_types(self, new: list[RelationTypeCard], known: list[RelationTypeCard]) -> Judgement: ...


class UnconfiguredLLMClient:
    """Keeps read-only endpoints usable when no API key is provided."""

    async def extract(self, text: str, known_relation_types: list[str]) -> ExtractionResult:
        raise LLMNotConfiguredError("OPENAI_API_KEY is not configured")

    async def analyze_image(self, image: bytes, mime_type: str, context: str) -> ImageAnalysis:
        raise LLMNotConfiguredError("OPENAI_API_KEY is not configured")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise LLMNotConfiguredError("OPENAI_API_KEY is not configured")

    async def judge_characters(self, new: list[CharacterCard], known: list[CharacterCard]) -> Judgement:
        raise LLMNotConfiguredError("OPENAI_API_KEY is not configured")

    async def judge_relation_types(self, new: list[RelationTypeCard], known: list[RelationTypeCard]) -> Judgement:
        raise LLMNotConfiguredError("OPENAI_API_KEY is not configured")

import asyncio
import base64
import logging
from typing import TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.llm import prompts
from app.llm.base import LLMError
from app.llm.judging import CharacterCard, RelationTypeCard
from app.llm.schemas import ExtractionResult, ImageAnalysis, Judgement

ResponseT = TypeVar("ResponseT", bound=BaseModel)

logger = logging.getLogger(__name__)

EMBEDDING_BATCH_SIZE = 256


def _lines(lines: list[str]) -> str:
    return "\n".join(lines) or "none"


def _candidates(refs: tuple[str, ...]) -> str:
    return f" | candidates: {', '.join(refs) or 'none'}"


def _known_character(card: CharacterCard) -> str:
    aliases = ", ".join(card.aliases) or "none"
    return f"{card.ref} | {card.name} | aliases: {aliases} | {card.description or 'no description'}"


def _new_character(card: CharacterCard) -> str:
    return _known_character(card) + _candidates(card.candidates)


def _new_relation_type(card: RelationTypeCard) -> str:
    return f"{card.ref} | {card.name}" + _candidates(card.candidates)


class OpenAILLMClient:
    """OpenAI (or compatible) backend. All calls share one semaphore to bound concurrency and rate usage."""

    def __init__(
        self,
        sdk: AsyncOpenAI,
        *,
        llm_model: str,
        vision_model: str,
        judge_model: str,
        embedding_model: str,
        embedding_dimensions: int,
        concurrency: int,
    ) -> None:
        self._sdk = sdk
        self._llm_model = llm_model
        self._vision_model = vision_model
        self._judge_model = judge_model
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._semaphore = asyncio.Semaphore(concurrency)

    async def extract(self, text: str, known_relation_types: list[str]) -> ExtractionResult:
        user_prompt = prompts.EXTRACTION_USER_TEMPLATE.format(
            known_relation_types=", ".join(known_relation_types) or "none yet",
            text=text,
        )
        return await self._parse(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": prompts.EXTRACTION_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            response_format=ExtractionResult,
            truncation_hint="; reduce CHUNK_SIZE_CHARS",
        )

    async def analyze_image(self, image: bytes, mime_type: str, context: str) -> ImageAnalysis:
        data_url = f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"
        return await self._parse(
            model=self._vision_model,
            messages=[
                {"role": "system", "content": prompts.VISION_SYSTEM_PROMPT.strip()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts.VISION_USER_TEMPLATE.format(context=context).strip()},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    ],
                },
            ],
            response_format=ImageAnalysis,
        )

    async def judge_characters(self, new: list[CharacterCard], known: list[CharacterCard]) -> Judgement:
        user_prompt = prompts.CHARACTER_JUDGE_USER_TEMPLATE.format(
            known=_lines([_known_character(card) for card in known]),
            new=_lines([_new_character(card) for card in new]),
        )
        return await self._parse(
            model=self._judge_model,
            messages=[
                {"role": "system", "content": prompts.CHARACTER_JUDGE_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            response_format=Judgement,
        )

    async def judge_relation_types(self, new: list[RelationTypeCard], known: list[RelationTypeCard]) -> Judgement:
        user_prompt = prompts.RELATION_TYPE_JUDGE_USER_TEMPLATE.format(
            known=_lines([f"{card.ref} | {card.name}" for card in known]),
            new=_lines([_new_relation_type(card) for card in new]),
        )
        return await self._parse(
            model=self._judge_model,
            messages=[
                {"role": "system", "content": prompts.RELATION_TYPE_JUDGE_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            response_format=Judgement,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            async with self._semaphore:
                try:
                    response = await self._sdk.embeddings.create(
                        model=self._embedding_model, input=batch, dimensions=self._embedding_dimensions
                    )
                except openai.OpenAIError as exc:
                    raise LLMError(f"embedding request failed: {exc}") from exc
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return vectors

    async def _parse(
        self, *, model: str, messages: list[dict], response_format: type[ResponseT], truncation_hint: str = ""
    ) -> ResponseT:
        async with self._semaphore:
            try:
                completion = await self._sdk.chat.completions.parse(
                    model=model, messages=messages, response_format=response_format, temperature=0
                )
            except openai.LengthFinishReasonError as exc:
                raise LLMError(f"{model} output was truncated{truncation_hint}") from exc
            except openai.OpenAIError as exc:
                raise LLMError(f"{model} request failed: {exc}") from exc
            except ValidationError as exc:
                raise LLMError(f"{model} returned malformed output: {exc}") from exc

        usage = getattr(completion, "usage", None)
        if usage is not None:
            logger.info("%s used %s prompt + %s completion tokens", model, usage.prompt_tokens, usage.completion_tokens)

        if not completion.choices:
            raise LLMError(f"{model} returned no choices")
        message = completion.choices[0].message
        if message.parsed is None:
            raise LLMError(f"{model} refused to answer: {message.refusal or 'empty response'}")
        return message.parsed


def build_openai_client(settings) -> OpenAILLMClient:
    sdk = AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    return OpenAILLMClient(
        sdk,
        llm_model=settings.llm_model,
        vision_model=settings.vision_model,
        judge_model=settings.judge_model,
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
        concurrency=settings.llm_concurrency,
    )

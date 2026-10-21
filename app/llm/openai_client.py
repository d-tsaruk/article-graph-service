import asyncio
import base64
from typing import TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.llm import prompts
from app.llm.base import LLMError
from app.llm.schemas import ExtractionResult, ImageAnalysis

ResponseT = TypeVar("ResponseT", bound=BaseModel)

EMBEDDING_BATCH_SIZE = 256


class OpenAILLMClient:
    """OpenAI (or compatible) backend. All calls share one semaphore to bound concurrency and rate usage."""

    def __init__(
        self,
        sdk: AsyncOpenAI,
        *,
        llm_model: str,
        vision_model: str,
        embedding_model: str,
        embedding_dimensions: int,
        concurrency: int,
    ) -> None:
        self._sdk = sdk
        self._llm_model = llm_model
        self._vision_model = vision_model
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

    async def _parse(self, *, model: str, messages: list[dict], response_format: type[ResponseT]) -> ResponseT:
        async with self._semaphore:
            try:
                completion = await self._sdk.chat.completions.parse(
                    model=model, messages=messages, response_format=response_format, temperature=0
                )
            except openai.LengthFinishReasonError as exc:
                raise LLMError("model output was truncated; reduce CHUNK_SIZE_CHARS") from exc
            except openai.OpenAIError as exc:
                raise LLMError(f"{model} request failed: {exc}") from exc

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
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
        concurrency=settings.llm_concurrency,
    )

"""Schemas passed to the LLM as Structured Outputs. Strict mode requires every field to be required."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExtractedCharacter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Most complete proper name of the character as used in the text")
    aliases: list[str] = Field(
        description="Other names, nicknames, titles or short forms that refer to the same character"
    )
    description: str = Field(description="One or two sentences: who the character is, based only on the text")


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="`name` of the character the relation goes from")
    target: str = Field(description="`name` of the character the relation goes to")
    type: str = Field(description="Relation type in UPPER_SNAKE_CASE, phrased so that `source TYPE target` reads")
    description: str = Field(description="Short justification of the relation grounded in the text")


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    characters: list[ExtractedCharacter]
    relations: list[ExtractedRelation]


class ImageAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_relevant: bool = Field(
        description="False for decorative content: logos, ornaments, page backgrounds, blank or unreadable images"
    )
    description: str = Field(description="Factual description of the visual content and any text it contains")


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str = Field(description="Reference of the new item being judged, e.g. N1")
    reason: str = Field(description="One short sentence explaining the decision, written before deciding")
    decision: Literal["stored", "same_as_new", "new"] = Field(
        description="stored: same as a known item; same_as_new: same as another new item; new: neither"
    )
    match: str | None = Field(description="S reference for `stored`, N reference for `same_as_new`, null for `new`")


class Judgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[Verdict] = Field(description="Exactly one verdict per new item")

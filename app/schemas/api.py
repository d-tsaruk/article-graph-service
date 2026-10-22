from typing import Literal

from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    app_token: str = Field(min_length=1)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Access token lifetime, seconds")
    refresh_expires_in: int = Field(description="Refresh token lifetime, seconds")


class CharacterOut(BaseModel):
    id: str
    name: str
    aliases: list[str]
    description: str


class RelationOut(BaseModel):
    source_id: str
    source_name: str
    target_id: str
    target_name: str
    type: str
    description: str


class ArticleUploadResponse(BaseModel):
    article_id: str
    title: str
    source_type: Literal["markdown", "pdf"]
    duplicate: bool = Field(description="True when identical content was already processed")
    characters: list[CharacterOut]
    relations: list[RelationOut]


class ArticleOut(BaseModel):
    article_id: str
    title: str
    source_type: Literal["markdown", "pdf"]
    status: Literal["processing", "stale", "completed"] = Field(
        description="`stale`: the upload that claimed this article died; upload the content again"
    )
    characters: list[CharacterOut]
    relations: list[RelationOut]


class RelationTypeOut(BaseModel):
    type: str
    count: int


class ConnectionOut(BaseModel):
    character: CharacterOut
    direction: Literal["outgoing", "incoming"] = Field(
        description="outgoing: `{character_id} TYPE other`; incoming: `other TYPE {character_id}`"
    )
    description: str


class ConnectedCharactersResponse(BaseModel):
    character_id: str
    requested_type: str
    resolved_type: str | None = Field(description="Stored relation type the request was matched to, if any")
    connections: list[ConnectionOut]


class ErrorResponse(BaseModel):
    detail: str


class ConflictResponse(ErrorResponse):
    article_id: str

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings
from app.core.security import InvalidTokenError, TokenService, TokenType
from app.graph.repository import GraphRepository
from app.pipeline.ingestion import ArticleIngestionService
from app.pipeline.resolution import RelationTypeResolver


@dataclass(frozen=True)
class Services:
    settings: Settings
    tokens: TokenService
    repository: GraphRepository
    ingestion: ArticleIngestionService
    relation_types: RelationTypeResolver


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]

_bearer = HTTPBearer(auto_error=False)


def unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail=detail, headers={"WWW-Authenticate": "Bearer"})


async def require_access_token(
    services: ServicesDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    if credentials is None:
        raise unauthorized("Missing bearer token")
    try:
        return services.tokens.verify(credentials.credentials, TokenType.ACCESS)
    except InvalidTokenError as exc:
        raise unauthorized("Invalid or expired access token") from exc

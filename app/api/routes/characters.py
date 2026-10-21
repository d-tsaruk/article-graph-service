from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import ServicesDep, require_access_token
from app.schemas.api import CharacterOut

router = APIRouter(prefix="/characters", tags=["characters"], dependencies=[Depends(require_access_token)])


@router.get("", summary="List all extracted characters")
async def list_characters(
    services: ServicesDep,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[CharacterOut]:
    characters = await services.repository.list_characters(limit=limit, offset=offset)
    return [CharacterOut(**vars(c)) for c in characters]


@router.get("/by_article/{article_id}", summary="List characters mentioned in an article")
async def characters_by_article(article_id: str, services: ServicesDep) -> list[CharacterOut]:
    characters = await services.repository.get_article_characters(article_id)
    if characters is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Article not found")
    return [CharacterOut(**vars(c)) for c in characters]

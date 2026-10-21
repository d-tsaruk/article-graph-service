from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import ServicesDep, require_access_token
from app.graph.repository import Direction
from app.schemas.api import CharacterOut, ConnectedCharactersResponse, ConnectionOut, RelationTypeOut

router = APIRouter(prefix="/relations", tags=["relations"], dependencies=[Depends(require_access_token)])


@router.get("/types", summary="List unique relation types with usage counts")
async def relation_types(services: ServicesDep) -> list[RelationTypeOut]:
    return [RelationTypeOut(type=t, count=c) for t, c in await services.repository.list_relation_types()]


@router.get(
    "/connected/{character_id}",
    summary="Characters connected to the given one by a relation type",
    description=(
        "`type` is normalized to UPPER_SNAKE_CASE (`friend of` -> `FRIEND_OF`). If no stored type matches "
        "exactly, the semantically closest one is used (vector index), see `resolved_type`."
    ),
)
async def connected_characters(
    character_id: str,
    services: ServicesDep,
    relation_type: Annotated[str, Query(alias="type", min_length=1, max_length=100)],
    direction: Direction = "any",
) -> ConnectedCharactersResponse:
    try:
        resolved = await services.relation_types.find(relation_type)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    # An unknown type still must 404 for an unknown character, so the lookup runs either way.
    connections = await services.repository.get_connections(character_id, resolved or "", direction)
    if connections is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Character not found")
    return ConnectedCharactersResponse(
        character_id=character_id,
        requested_type=relation_type,
        resolved_type=resolved,
        connections=[
            ConnectionOut(character=CharacterOut(**vars(c.character)), direction=c.direction, description=c.description)
            for c in connections
        ],
    )

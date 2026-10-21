from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import ServicesDep, require_access_token
from app.graph.repository import Direction
from app.pipeline.normalization import normalize_relation_type
from app.schemas.api import CharacterOut, ConnectedCharactersResponse, ConnectionOut, RelationTypeOut

_FLIPPED: dict[str, Direction] = {"outgoing": "incoming", "incoming": "outgoing", "any": "any"}

router = APIRouter(prefix="/relations", tags=["relations"], dependencies=[Depends(require_access_token)])


@router.get("/types", summary="List unique relation types with usage counts")
async def relation_types(services: ServicesDep) -> list[RelationTypeOut]:
    return [RelationTypeOut(type=t, count=c) for t, c in await services.repository.list_relation_types()]


@router.get(
    "/connected/{character_id}",
    summary="Characters connected to the given one by a relation type",
    description=(
        "`type` is normalized to UPPER_SNAKE_CASE (`friend of` -> `FRIEND_OF`). Resolution order: "
        "1. a known inverse such as `child of`, answered through the canonical type (`PARENT_OF`) with the "
        "direction flipped, even if a stored type of that name exists; "
        "2. exact stored type; "
        "3. vector candidates confirmed by the LLM judge. "
        "`resolved_type` shows the type used. With the default `direction=any`, an inverse type also returns "
        "the connections in the other direction (for example a character's children for `child of`); "
        "each connection's `direction` field tells them apart."
    ),
)
async def connected_characters(
    character_id: str,
    services: ServicesDep,
    relation_type: Annotated[str, Query(alias="type", min_length=1, max_length=100)],
    direction: Direction = "any",
) -> ConnectedCharactersResponse:
    try:
        normalized = normalize_relation_type(relation_type)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    # Outside the try: a backend failure is an LLMError (502), never a client error.
    query = await services.relation_types.find(normalized)
    resolved = query.name if query else None
    effective_direction = _FLIPPED[direction] if query and query.inverse else direction

    # An unknown type still must 404 for an unknown character, so the lookup runs either way.
    connections = await services.repository.get_connections(character_id, resolved or "", effective_direction)
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

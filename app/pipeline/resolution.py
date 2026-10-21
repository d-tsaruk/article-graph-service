"""Maps freshly extracted entities onto what the graph already knows, using the vector indexes."""

import logging
import uuid

from app.graph.repository import CharacterMatch, CharacterRecord, GraphRepository
from app.llm.base import LLMClient
from app.llm.schemas import ExtractedCharacter
from app.pipeline.merging import MAX_DESCRIPTION_CHARS
from app.pipeline.normalization import name_key, names_compatible, normalize_relation_type, relation_type_phrase

logger = logging.getLogger(__name__)

SIMILAR_CANDIDATES = 5


def character_embedding_text(character: ExtractedCharacter) -> str:
    aliases = f" (also known as {', '.join(character.aliases)})" if character.aliases else ""
    return f"{character.name}{aliases}: {character.description}"


def _unique_by_key(names: list[str], exclude_key: str) -> tuple[list[str], list[str]]:
    seen = {exclude_key}
    unique: list[str] = []
    keys: list[str] = []
    for name in names:
        key = name_key(name)
        if key and key not in seen:
            seen.add(key)
            unique.append(name.strip())
            keys.append(key)
    return unique, keys


class CharacterResolver:
    def __init__(
        self, repository: GraphRepository, llm: LLMClient, *, match_score: float, auto_merge_score: float
    ) -> None:
        self._repository = repository
        self._llm = llm
        self._match_score = match_score
        self._auto_merge_score = auto_merge_score

    async def resolve(self, characters: list[ExtractedCharacter]) -> dict[str, CharacterRecord]:
        """Returns extracted name -> stored character. Processed sequentially on purpose: a character
        created for this article must be visible to the vector index when the next one is resolved."""
        embeddings = await self._llm.embed([character_embedding_text(c) for c in characters])
        resolved: dict[str, CharacterRecord] = {}
        for character, embedding in zip(characters, embeddings, strict=True):
            resolved[character.name] = await self._resolve_one(character, embedding)
        return resolved

    async def _resolve_one(self, character: ExtractedCharacter, embedding: list[float]) -> CharacterRecord:
        names = [character.name, *character.aliases]
        existing = await self._repository.find_character_by_keys([name_key(n) for n in names])
        if existing is None:
            candidates = await self._repository.find_similar_characters(embedding, SIMILAR_CANDIDATES)
            existing = self._pick_candidate(candidates, names)

        if existing is None:
            return await self._create(character, embedding)
        return await self._merge(existing, character)

    def _pick_candidate(self, candidates: list[CharacterMatch], names: list[str]) -> CharacterRecord | None:
        for candidate in candidates:
            stored = candidate.character
            if candidate.score >= self._auto_merge_score or (
                candidate.score >= self._match_score and names_compatible(names, [stored.name, *stored.aliases])
            ):
                logger.info("Merging %r into %r (score %.3f)", names[0], stored.name, candidate.score)
                return stored
        return None

    async def _create(self, character: ExtractedCharacter, embedding: list[float]) -> CharacterRecord:
        aliases, alias_keys = _unique_by_key(character.aliases, exclude_key=name_key(character.name))
        record = CharacterRecord(
            id=str(uuid.uuid4()), name=character.name, aliases=aliases, description=character.description
        )
        await self._repository.create_character(
            record, name_key=name_key(character.name), alias_keys=alias_keys, embedding=embedding
        )
        return record

    async def _merge(self, existing: CharacterRecord, character: ExtractedCharacter) -> CharacterRecord:
        aliases, alias_keys = _unique_by_key(
            [*existing.aliases, character.name, *character.aliases], exclude_key=name_key(existing.name)
        )
        description = existing.description
        if character.description and character.description not in description:
            description = f"{description} {character.description}".strip()[:MAX_DESCRIPTION_CHARS]
        merged = CharacterRecord(id=existing.id, name=existing.name, aliases=aliases, description=description)
        if merged != existing:
            await self._repository.update_character(merged, alias_keys=alias_keys)
        return merged


class RelationTypeResolver:
    def __init__(self, repository: GraphRepository, llm: LLMClient, *, match_score: float) -> None:
        self._repository = repository
        self._llm = llm
        self._match_score = match_score

    async def find(self, relation_type: str) -> str | None:
        """Exact name first, then the closest stored type by meaning. Raises ValueError for empty input."""
        normalized = normalize_relation_type(relation_type)
        if await self._repository.relation_type_exists(normalized):
            return normalized
        return await self._nearest(normalized, await self._embed(normalized))

    async def resolve_or_create(self, relation_type: str) -> str:
        normalized = normalize_relation_type(relation_type)
        if await self._repository.relation_type_exists(normalized):
            return normalized
        embedding = await self._embed(normalized)
        if match := await self._nearest(normalized, embedding):
            return match
        await self._repository.create_relation_type(normalized, embedding)
        return normalized

    async def _embed(self, normalized: str) -> list[float]:
        [embedding] = await self._llm.embed([relation_type_phrase(normalized)])
        return embedding

    async def _nearest(self, normalized: str, embedding: list[float]) -> str | None:
        matches = await self._repository.find_similar_relation_types(embedding, limit=1)
        if matches and matches[0].score >= self._match_score:
            logger.info("Relation type %s resolved to %s (score %.3f)", normalized, matches[0].name, matches[0].score)
            return matches[0].name
        return None

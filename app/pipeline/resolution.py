"""Maps freshly extracted entities onto what the graph already knows: retrieval proposes, an LLM judge decides."""

import logging
import uuid
from dataclasses import dataclass

from app.graph.repository import CharacterRecord, GraphRepository
from app.llm.base import LLMClient, LLMError
from app.llm.judging import CharacterCard, RelationTypeCard
from app.llm.schemas import ExtractedCharacter
from app.llm.vocabulary import CANONICAL_RELATION_TYPES, INVERSE_RELATION_TYPES
from app.pipeline.adjudication import group_verdicts
from app.pipeline.merging import MAX_DESCRIPTION_CHARS
from app.pipeline.normalization import identifying_tokens, name_key, normalize_relation_type, relation_type_phrase

logger = logging.getLogger(__name__)

LEXICAL_CANDIDATES = 10
RELATION_TYPE_CANDIDATES = 5
# The vector index is approximate and `k` is also its search width: asking for more than we keep improves recall.
RELATION_TYPE_SEARCH_WIDTH = 4 * RELATION_TYPE_CANDIDATES


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


@dataclass
class CharacterPlan:
    exact: list[tuple[CharacterRecord, ExtractedCharacter]]
    merges: list[tuple[CharacterRecord, list[ExtractedCharacter]]]
    creations: list[tuple[ExtractedCharacter, list[float], list[ExtractedCharacter]]]  # canonical, embedding, others


async def _embed(llm: LLMClient, texts: list[str]) -> list[list[float]]:
    embeddings = await llm.embed(texts)
    if len(embeddings) != len(texts):
        raise LLMError(f"embedding backend returned {len(embeddings)} vectors for {len(texts)} texts")
    return embeddings


def _are_inverse(left: str, right: str) -> bool:
    return INVERSE_RELATION_TYPES.get(left) == right or INVERSE_RELATION_TYPES.get(right) == left


class CharacterResolver:
    """Candidates come from three channels (shared name or alias, shared name token, vector neighbours);
    one batched LLM call per upload decides which of them are the same individual."""

    def __init__(self, repository: GraphRepository, llm: LLMClient, *, candidate_limit: int) -> None:
        self._repository = repository
        self._llm = llm
        self._candidate_limit = candidate_limit

    async def plan(self, characters: list[ExtractedCharacter]) -> CharacterPlan:
        """Only reads the graph and calls the LLM; never writes."""
        plan = CharacterPlan(exact=[], merges=[], creations=[])
        if not characters:
            return plan
        embeddings = await _embed(self._llm, [character_embedding_text(c) for c in characters])

        pending: list[tuple[ExtractedCharacter, list[float]]] = []
        for character, embedding in zip(characters, embeddings, strict=True):
            stored = await self._repository.find_character_by_name_key(name_key(character.name))
            if stored is not None:
                plan.exact.append((stored, character))
            else:
                pending.append((character, embedding))
        if not pending:
            return plan

        found = [await self._candidates(character, embedding) for character, embedding in pending]
        records = {record.id: record for candidates in found for record in candidates}
        known_refs = {record_id: f"S{i}" for i, record_id in enumerate(records, 1)}
        new_refs = [f"N{i}" for i in range(1, len(pending) + 1)]
        candidates = {
            ref: tuple(known_refs[record.id] for record in own) for ref, own in zip(new_refs, found, strict=True)
        }

        judgement = None
        if records or len(pending) > 1:
            judgement = await self._llm.judge_characters(
                [
                    CharacterCard(ref, c.name, tuple(c.aliases), c.description, candidates[ref])
                    for ref, (c, _) in zip(new_refs, pending, strict=True)
                ],
                [CharacterCard(known_refs[r.id], r.name, tuple(r.aliases), r.description) for r in records.values()],
            )

        by_ref = dict(zip(new_refs, pending, strict=True))
        record_by_ref = {ref: records[record_id] for record_id, ref in known_refs.items()}
        labels = {ref: c.name for ref, (c, _) in by_ref.items()} | {ref: r.name for ref, r in record_by_ref.items()}
        for group in group_verdicts(candidates, judgement, kind="character", labels=labels):
            members = [by_ref[ref][0] for ref in group.members]
            if group.stored:
                record = record_by_ref[group.stored]
                logger.info("Merging %s into %r", [m.name for m in members], record.name)
                plan.merges.append((record, members))
            else:
                canonical, embedding = by_ref[group.canonical]
                others = [m for m in members if m is not canonical]
                plan.creations.append((canonical, embedding, others))
        return plan

    async def apply(self, plan: CharacterPlan) -> dict[str, CharacterRecord]:
        """Writes the plan and returns extracted name -> stored character. Makes no LLM call."""
        resolved: dict[str, CharacterRecord] = {}
        current: dict[str, CharacterRecord] = {}  # by id, so that merges into one stored character build on each other
        for stored, members in [*((record, [member]) for record, member in plan.exact), *plan.merges]:
            for member in members:
                record = await self._merge(current.get(stored.id, stored), member)
                current[stored.id] = record
                resolved[member.name] = record
        for canonical, embedding, others in plan.creations:
            record = await self._create(canonical, embedding)
            for other in others:
                record = await self._merge(record, other)
            for member in (canonical, *others):
                resolved[member.name] = record
        return resolved

    async def _candidates(self, character: ExtractedCharacter, embedding: list[float]) -> list[CharacterRecord]:
        names = [character.name, *character.aliases]
        channels = [
            await self._repository.find_characters_by_keys([name_key(n) for n in names]),
            await self._repository.find_characters_by_tokens(sorted(identifying_tokens(names)), LEXICAL_CANDIDATES),
            [m.character for m in await self._repository.find_similar_characters(embedding, self._candidate_limit)],
        ]
        return list({record.id: record for channel in channels for record in channel}.values())

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


@dataclass(frozen=True)
class TypePlan:
    mapping: dict[str, str]  # extracted normalized type -> type to store
    to_create: dict[str, list[float]]  # types to store that do not exist yet -> embedding


@dataclass(frozen=True)
class TypeQuery:
    name: str
    inverse: bool  # True when answered through INVERSE_RELATION_TYPES: the caller flips the direction


class RelationTypeResolver:
    """Keeps the relation-type vocabulary free of synonyms: the vector index proposes candidates, the LLM decides."""

    def __init__(self, repository: GraphRepository, llm: LLMClient, *, candidate_score: float) -> None:
        self._repository = repository
        self._llm = llm
        self._candidate_score = candidate_score
        self._vocabulary_ready = False

    async def ensure_vocabulary(self) -> None:
        if self._vocabulary_ready:
            return
        present = await self._repository.existing_relation_types(list(CANONICAL_RELATION_TYPES))
        missing = [name for name in CANONICAL_RELATION_TYPES if name not in present]
        if missing:
            embeddings = await _embed(self._llm, [relation_type_phrase(name) for name in missing])
            await self._repository.create_relation_types(list(zip(missing, embeddings, strict=True)))
        self._vocabulary_ready = True

    async def plan(self, types: list[str]) -> TypePlan:
        """Maps every extracted type to the one to store. Only reads and LLM calls: nothing is written
        except the idempotent vocabulary seed."""
        await self.ensure_vocabulary()
        distinct = list(dict.fromkeys(types))
        existing = await self._repository.existing_relation_types(distinct)
        new = [name for name in distinct if name not in existing]
        mapping = {name: name for name in existing}
        if not new:
            return TypePlan(mapping, {})

        embeddings = await _embed(self._llm, [relation_type_phrase(name) for name in new])
        neighbours = [
            await self._candidates(embedding, exclude_inverse_of=name)
            for name, embedding in zip(new, embeddings, strict=True)
        ]
        known_refs = {
            name: f"S{i}" for i, name in enumerate(dict.fromkeys(n for found in neighbours for n in found), 1)
        }
        new_refs = {name: f"N{i}" for i, name in enumerate(new, 1)}
        candidates = {
            new_refs[name]: tuple(known_refs[n] for n in found) for name, found in zip(new, neighbours, strict=True)
        }

        judgement = None
        if known_refs or len(new) > 1:
            judgement = await self._llm.judge_relation_types(
                [RelationTypeCard(new_refs[name], name, candidates[new_refs[name]]) for name in new],
                [RelationTypeCard(ref, name) for name, ref in known_refs.items()],
            )

        known_by_ref = {ref: name for name, ref in known_refs.items()}
        new_by_ref = {ref: name for name, ref in new_refs.items()}
        embedding_of = dict(zip(new, embeddings, strict=True))
        to_create: dict[str, list[float]] = {}
        labels = {**{ref: name for name, ref in new_refs.items()}, **{ref: name for name, ref in known_refs.items()}}
        for group in group_verdicts(candidates, judgement, kind="relation type", labels=labels):
            if group.stored:
                target = known_by_ref[group.stored]
            else:
                target = new_by_ref[group.canonical]
                to_create[target] = embedding_of[target]
            for member in group.members:
                mapping[new_by_ref[member]] = target
                if target != new_by_ref[member]:
                    logger.info("Relation type %s resolved to %s", new_by_ref[member], target)
        return TypePlan(mapping, to_create)

    async def create_used(self, plan: TypePlan, used: set[str]) -> None:
        """Creates only the planned types that a stored edge actually uses."""
        await self._repository.create_relation_types(
            [(name, embedding) for name, embedding in plan.to_create.items() if name in used]
        )

    async def find(self, relation_type: str) -> TypeQuery | None:
        """Query side: a known inverse, then exact name, then vector candidates confirmed by the judge.
        The inverse map comes first so that a stray stored `CHILD_OF` cannot hide the `PARENT_OF` edges.
        Raises ValueError for empty input."""
        normalized = normalize_relation_type(relation_type)
        if normalized in INVERSE_RELATION_TYPES:
            return TypeQuery(INVERSE_RELATION_TYPES[normalized], inverse=True)
        if normalized in CANONICAL_RELATION_TYPES or await self._repository.existing_relation_types([normalized]):
            return TypeQuery(normalized, inverse=False)

        await self.ensure_vocabulary()
        [embedding] = await _embed(self._llm, [relation_type_phrase(normalized)])
        found = await self._candidates(embedding, exclude_inverse_of=normalized)
        if not found:
            return None
        refs = {name: f"S{i}" for i, name in enumerate(found, 1)}
        judgement = await self._llm.judge_relation_types(
            [RelationTypeCard("N1", normalized, tuple(refs.values()))],
            [RelationTypeCard(ref, name) for name, ref in refs.items()],
        )
        [group] = group_verdicts(
            {"N1": tuple(refs.values())},
            judgement,
            kind="relation type",
            labels={"N1": normalized, **{ref: name for name, ref in refs.items()}},
        )
        if group.stored is None:
            return None
        stored = next(name for name, ref in refs.items() if ref == group.stored)
        logger.info("Relation type %s resolved to %s", normalized, stored)
        return TypeQuery(stored, inverse=False)

    async def _candidates(self, embedding: list[float], *, exclude_inverse_of: str) -> list[str]:
        """An inverse is never a match candidate (spec D7), whatever the judge would say."""
        matches = await self._repository.find_similar_relation_types(embedding, RELATION_TYPE_SEARCH_WIDTH)
        names = [
            match.name
            for match in matches
            if match.score >= self._candidate_score and not _are_inverse(exclude_inverse_of, match.name)
        ]
        return names[:RELATION_TYPE_CANDIDATES]

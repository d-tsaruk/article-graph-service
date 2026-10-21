import logging
from dataclasses import dataclass, field

from app.llm.schemas import ExtractedCharacter, ExtractedRelation, ExtractionResult
from app.llm.vocabulary import INVERSE_RELATION_TYPES
from app.pipeline.normalization import name_key, normalize_relation_type

logger = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 1000


@dataclass
class _Group:
    name: str
    names: dict[str, str] = field(default_factory=dict)  # name_key -> first seen spelling
    descriptions: list[str] = field(default_factory=list)

    def add_name(self, name: str) -> None:
        key = name_key(name)
        if key:
            self.names.setdefault(key, name.strip())

    def add_description(self, description: str) -> None:
        description = description.strip()
        if description and description not in self.descriptions:
            self.descriptions.append(description)

    def absorb(self, other: "_Group") -> None:
        for key, spelling in other.names.items():
            self.names.setdefault(key, spelling)
        for description in other.descriptions:
            self.add_description(description)

    def to_character(self) -> ExtractedCharacter:
        canonical = name_key(self.name)
        aliases = [spelling for key, spelling in self.names.items() if key != canonical]
        description = " ".join(self.descriptions)[:MAX_DESCRIPTION_CHARS]
        return ExtractedCharacter(name=self.name, aliases=aliases, description=description)


class _CharacterIndex:
    """Union-find over name keys: entries sharing any name or alias end up in one group."""

    def __init__(self) -> None:
        self._groups: list[_Group] = []
        self._parent: list[int] = []
        self._by_key: dict[str, int] = {}

    def _find(self, index: int) -> int:
        while self._parent[index] != index:
            self._parent[index] = self._parent[self._parent[index]]
            index = self._parent[index]
        return index

    def add(self, character: ExtractedCharacter) -> None:
        keys = {name_key(n) for n in (character.name, *character.aliases)} - {""}
        if not keys:
            return
        roots = sorted({self._find(self._by_key[k]) for k in keys if k in self._by_key})
        if roots:
            # The earliest group wins so the canonical name is the first one the model produced.
            target = roots[0]
            for other in roots[1:]:
                self._groups[target].absorb(self._groups[other])
                self._parent[other] = target
        else:
            target = len(self._groups)
            self._groups.append(_Group(name=character.name.strip()))
            self._parent.append(target)

        group = self._groups[target]
        for name in (character.name, *character.aliases):
            group.add_name(name)
        group.add_description(character.description)
        for key in keys:
            self._by_key[key] = target

    def canonical_name(self, name: str) -> str | None:
        index = self._by_key.get(name_key(name))
        return None if index is None else self._groups[self._find(index)].name

    def characters(self) -> list[ExtractedCharacter]:
        return [group.to_character() for i, group in enumerate(self._groups) if self._find(i) == i]


def merge_extractions(results: list[ExtractionResult]) -> ExtractionResult:
    """Combines per-chunk extractions of one article into a single consistent result."""
    index = _CharacterIndex()
    for result in results:
        for character in result.characters:
            index.add(character)

    relations: dict[tuple[str, str, str], ExtractedRelation] = {}
    for result in results:
        for relation in result.relations:
            source = index.canonical_name(relation.source)
            target = index.canonical_name(relation.target)
            if source is None or target is None:
                logger.info("Dropping relation to unknown character: %s", relation)
                continue
            if source == target:
                continue
            try:
                relation_type = normalize_relation_type(relation.type)
            except ValueError:
                logger.info("Dropping relation with empty type: %s", relation)
                continue
            if relation_type in INVERSE_RELATION_TYPES:
                source, target = target, source
                relation_type = INVERSE_RELATION_TYPES[relation_type]
            relations.setdefault(
                (source, target, relation_type),
                ExtractedRelation(
                    source=source, target=target, type=relation_type, description=relation.description.strip()
                ),
            )

    return ExtractionResult(characters=index.characters(), relations=list(relations.values()))

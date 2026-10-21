import logging
import re
from dataclasses import dataclass, field

from app.llm.schemas import ExtractedCharacter, ExtractedRelation, ExtractionResult
from app.llm.vocabulary import INVERSE_RELATION_TYPES
from app.pipeline.normalization import name_key, normalize_relation_type

logger = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 1000
POSSESSIVE = re.compile(r"['’]s\b")


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


def _clean_aliases(character: ExtractedCharacter) -> list[tuple[str, str]]:
    """(spelling, key) pairs of the usable aliases: no possessive descriptors, no echo of the name."""
    name_k = name_key(character.name)
    aliases = []
    for alias in character.aliases:
        if POSSESSIVE.search(alias):
            logger.info("Dropping possessive descriptor alias %r of %r", alias, character.name)
        elif (alias_k := name_key(alias)) and alias_k != name_k:
            aliases.append((alias, alias_k))
    return aliases


class _CharacterIndex:
    """Groups characters anchored on names.

    Two entries are unioned only when the name of one equals the name or an alias of the other. An alias shared
    only alias-to-alias ("Voldemort" for both Tom Riddle and Lord Voldemort) never unions: that call is left to
    the judge, because models also emit descriptors such as "Ronald's father" as aliases of the wrong person.
    An alias held by characters with different names is ambiguous for the whole article, so it neither unions
    nor resolves a relation, whatever the order the characters arrive in.
    """

    def __init__(self, characters: list[ExtractedCharacter]) -> None:
        self._groups: list[_Group] = []
        self._parent: list[int] = []
        self._by_name_key: dict[str, int] = {}
        self._by_alias_key: dict[str, set[int]] = {}
        holders: dict[str, set[str]] = {}
        self._cleaned = []
        for character in characters:
            name_k = name_key(character.name)
            if not name_k:
                continue
            aliases = _clean_aliases(character)
            self._cleaned.append((character, name_k, aliases))
            for _, alias_k in aliases:
                holders.setdefault(alias_k, set()).add(name_k)
        self._ambiguous = {key for key, names in holders.items() if len(names) > 1}
        for character, name_k, aliases in self._cleaned:
            self._add(character, name_k, aliases)

    def _find(self, index: int) -> int:
        while self._parent[index] != index:
            self._parent[index] = self._parent[self._parent[index]]
            index = self._parent[index]
        return index

    def _add(self, character: ExtractedCharacter, name_k: str, aliases: list[tuple[str, str]]) -> None:
        matched = set()
        if name_k in self._by_name_key:
            matched.add(self._find(self._by_name_key[name_k]))
        if name_k not in self._ambiguous:
            matched.update(self._alias_roots(name_k))
        for _, alias_k in aliases:
            if alias_k in self._by_name_key and alias_k not in self._ambiguous:
                matched.add(self._find(self._by_name_key[alias_k]))

        if matched:
            # The earliest group wins so the canonical name is the first one the model produced.
            target = min(matched)
            for other in matched - {target}:
                self._groups[target].absorb(self._groups[other])
                self._parent[other] = target
        else:
            target = len(self._groups)
            self._groups.append(_Group(name=character.name.strip()))
            self._parent.append(target)

        group = self._groups[target]
        group.add_name(character.name)
        for alias, _ in aliases:
            group.add_name(alias)
        group.add_description(character.description)
        self._by_name_key[name_k] = target
        for _, alias_k in aliases:
            self._by_alias_key.setdefault(alias_k, set()).add(target)

    def _alias_roots(self, key: str) -> set[int]:
        return {self._find(i) for i in self._by_alias_key.get(key, ())}

    def canonical_name(self, name: str) -> str | None:
        key = name_key(name)
        if key in self._by_name_key:
            return self._groups[self._find(self._by_name_key[key])].name
        roots = self._alias_roots(key)
        if key in self._ambiguous or len(roots) != 1:
            return None
        return self._groups[roots.pop()].name

    def is_ambiguous(self, name: str) -> bool:
        key = name_key(name)
        return key not in self._by_name_key and key in self._ambiguous

    def characters(self) -> list[ExtractedCharacter]:
        return [group.to_character() for i, group in enumerate(self._groups) if self._find(i) == i]


def merge_extractions(results: list[ExtractionResult]) -> ExtractionResult:
    """Combines per-chunk extractions of one article into a single consistent result."""
    index = _CharacterIndex([character for result in results for character in result.characters])

    relations: dict[tuple[str, str, str], ExtractedRelation] = {}
    for result in results:
        for relation in result.relations:
            source = index.canonical_name(relation.source)
            target = index.canonical_name(relation.target)
            if source is None or target is None:
                ambiguous = [n for n in (relation.source, relation.target) if index.is_ambiguous(n)]
                if ambiguous:
                    logger.info("Dropping relation through ambiguous alias %r: %s", ambiguous[0], relation)
                else:
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

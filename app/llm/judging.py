from dataclasses import dataclass


@dataclass(frozen=True)
class CharacterCard:
    ref: str
    name: str
    aliases: tuple[str, ...]
    description: str
    candidates: tuple[str, ...] = ()  # S refs offered for a new character; empty for known ones


@dataclass(frozen=True)
class RelationTypeCard:
    ref: str
    name: str
    candidates: tuple[str, ...] = ()

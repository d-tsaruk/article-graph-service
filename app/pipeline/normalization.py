import re
import unicodedata
from collections.abc import Iterable

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)

# Tokens that carry no identity on their own: "Mr. Potter" and "Harry Potter" share only "potter".
_NON_IDENTIFYING_TOKENS = frozenset(
    {
        "mr", "mrs", "ms", "miss", "mister", "madam", "madame", "dr", "doctor", "prof", "professor",
        "sir", "lady", "lord", "dame", "master", "the", "of", "a", "an", "jr", "sr",
    }
)  # fmt: skip


def name_key(name: str) -> str:
    """Case-, width- and punctuation-insensitive key used for exact name matching."""
    normalized = unicodedata.normalize("NFKC", name).casefold()
    return " ".join(_NON_WORD.sub(" ", normalized).split())


def identifying_tokens(names: Iterable[str]) -> set[str]:
    return {
        token
        for name in names
        for token in name_key(name).split()
        if len(token) > 1 and token not in _NON_IDENTIFYING_TOKENS
    }


def normalize_relation_type(raw: str) -> str:
    normalized = unicodedata.normalize("NFKC", raw).upper()
    result = _NON_WORD.sub("_", normalized).strip("_")
    if not result:
        raise ValueError(f"relation type {raw!r} has no word characters")
    return result


def relation_type_phrase(relation_type: str) -> str:
    """Natural-language form embedded for semantic matching: FRIEND_OF -> 'friend of'."""
    return relation_type.replace("_", " ").lower()

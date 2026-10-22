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


def _full_names(names: Iterable[str]) -> list[tuple[str, str]]:
    """(given, surname) of every name that has at least two identifying tokens."""
    pairs = []
    for name in names:
        tokens = [t for t in name_key(name).split() if len(t) > 1 and t not in _NON_IDENTIFYING_TOKENS]
        if len(tokens) >= 2:
            pairs.append((tokens[0], tokens[-1]))
    return pairs


def names_conflict(left: Iterable[str], right: Iterable[str]) -> bool:
    """True when both sides carry full names of one family (same surname) and none of those pairs could be one
    person: "Percy Weasley" vs "Ron Weasley" conflict, "Ron Weasley" vs "Ronald Weasley" do not.
    No evidence, no veto."""
    same_surname = [
        (a_given, b_given)
        for a_given, a_surname in _full_names(left)
        for b_given, b_surname in _full_names(right)
        if a_surname == b_surname
    ]
    return bool(same_surname) and not any(a.startswith(b) or b.startswith(a) for a, b in same_surname)


def normalize_relation_type(raw: str) -> str:
    normalized = unicodedata.normalize("NFKC", raw).upper()
    result = _NON_WORD.sub("_", normalized).strip("_")
    if not result:
        raise ValueError(f"relation type {raw!r} has no word characters")
    return result


def relation_type_phrase(relation_type: str) -> str:
    """Natural-language form embedded for semantic matching: FRIEND_OF -> 'friend of'."""
    return relation_type.replace("_", " ").lower()

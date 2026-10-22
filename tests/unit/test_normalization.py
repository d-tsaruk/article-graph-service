import pytest

from app.pipeline.normalization import (
    identifying_tokens,
    name_key,
    names_conflict,
    normalize_relation_type,
    relation_type_phrase,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Harry  Potter", "harry potter"),
        ("  HARRY POTTER. ", "harry potter"),
        ("Hermione Jean-Granger", "hermione jean granger"),
        ("Ｈａｒｒｙ", "harry"),
    ],
)
def test_name_key(raw: str, expected: str) -> None:
    assert name_key(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("friend of", "FRIEND_OF"),
        ("Friend-Of", "FRIEND_OF"),
        ("  is   parent of ", "IS_PARENT_OF"),
        ("ENEMY_OF", "ENEMY_OF"),
        ("works for!", "WORKS_FOR"),
    ],
)
def test_normalize_relation_type(raw: str, expected: str) -> None:
    assert normalize_relation_type(raw) == expected


def test_normalize_relation_type_rejects_empty_values() -> None:
    with pytest.raises(ValueError):
        normalize_relation_type(" !! ")


def test_relation_type_phrase() -> None:
    assert relation_type_phrase("FRIEND_OF") == "friend of"


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["Mr. Dursley"], {"dursley"}),
        (["Harry Potter", "Harry"], {"harry", "potter"}),
        (["Mr."], set()),
        (["The Dark Lord"], {"dark"}),
    ],
)
def test_identifying_tokens(names: list[str], expected: set[str]) -> None:
    assert identifying_tokens(names) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (["Percy Weasley"], ["Ron Weasley", "Ronald"], True),
        (["Arthur Weasley"], ["Molly Weasley"], True),
        (["Ron Weasley"], ["Ronald Weasley"], False),
        (["Mr. Weasley"], ["Arthur Weasley"], False),
        (["Tom Riddle"], ["Lord Voldemort"], False),
        (["Harry Potter"], ["The Boy Who Lived"], False),
        (["Ron Weasley", "Percy Weasley"], ["Ron Weasley"], False),
    ],
)
def test_names_conflict(left: list[str], right: list[str], expected: bool) -> None:
    assert names_conflict(left, right) is expected

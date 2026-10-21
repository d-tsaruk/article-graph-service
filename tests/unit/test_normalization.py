import pytest

from app.pipeline.normalization import name_key, names_compatible, normalize_relation_type, relation_type_phrase


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
    ("left", "right", "expected"),
    [
        (["Harry Potter"], ["Harry"], True),
        (["Mr. Dursley"], ["Vernon Dursley"], True),
        (["Professor Dumbledore"], ["Albus Dumbledore"], True),
        (["Harry Potter"], ["Ron Weasley"], False),
        (["Mr. Smith"], ["Mrs. Jones"], False),
        (["Mr."], ["Mr."], False),
        (["The Dark Lord"], ["Voldemort", "Dark Lord"], True),
    ],
)
def test_names_compatible(left: list[str], right: list[str], expected: bool) -> None:
    assert names_compatible(left, right) is expected

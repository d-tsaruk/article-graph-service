import itertools
import logging
import re

import pytest

from app.pipeline import merging
from app.pipeline.merging import merge_extractions
from tests.factories import character, extraction, relation


def test_characters_from_different_chunks_are_merged_by_name_and_alias() -> None:
    merged = merge_extractions(
        [
            extraction([character("Harry Potter", "A wizard", aliases=("Harry",))]),
            extraction([character("Harry", "The boy with the scar", aliases=("The Boy Who Lived",))]),
            extraction([character("the boy who lived", "")]),
        ]
    )

    assert len(merged.characters) == 1
    harry = merged.characters[0]
    assert harry.name == "Harry Potter"
    assert set(harry.aliases) == {"Harry", "The Boy Who Lived"}
    assert harry.description == "A wizard The boy with the scar"


def test_alias_only_overlap_is_left_to_the_judge() -> None:
    merged = merge_extractions(
        [
            extraction([character("Tom Riddle", aliases=("Voldemort",))]),
            extraction([character("Lord Voldemort", aliases=("Voldemort", "You-Know-Who"))]),
        ]
    )

    assert [c.name for c in merged.characters] == ["Tom Riddle", "Lord Voldemort"]


RAW_FAMILY = [
    ("The Boy Who Lived", ("the orphan raised by the Dursleys", "the boy")),
    ("Ronald", ("Ronald Weasley", "Ronald's father", "Ronald's mother", "Ronald's older brother")),
    ("Mr. Weasley", ("Ronald's father",)),
    ("Mrs. Weasley", ("Ronald's mother",)),
    ("Percy Weasley", ("Ronald's older brother", "Percy")),
    ("Tom Riddle", ("Lord Voldemort",)),
]
FAMILY_NAMES = ["The Boy Who Lived", "Ronald", "Mr. Weasley", "Mrs. Weasley", "Percy Weasley", "Tom Riddle"]


def test_descriptor_aliases_do_not_collapse_a_family() -> None:
    merged = merge_extractions([extraction([character(name, aliases=aliases) for name, aliases in RAW_FAMILY])])

    assert [c.name for c in merged.characters] == FAMILY_NAMES


def test_name_anchoring_alone_keeps_a_family_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(merging, "POSSESSIVE", re.compile(r"(?!)"))

    merged = merge_extractions([extraction([character(name, aliases=aliases) for name, aliases in RAW_FAMILY])])

    assert [c.name for c in merged.characters] == FAMILY_NAMES


@pytest.mark.parametrize(
    "order",
    list(
        itertools.permutations(
            [
                character("Arthur Weasley", aliases=("Mr Weasley",)),
                character("Percy Weasley", aliases=("Mr Weasley",)),
                character("Mr Weasley"),
            ]
        )
    ),
)
def test_name_matching_an_ambiguous_alias_merges_with_neither_group(order) -> None:
    merged = merge_extractions([extraction(list(order))])

    assert sorted(c.name for c in merged.characters) == ["Arthur Weasley", "Mr Weasley", "Percy Weasley"]


def test_ambiguity_is_judged_across_chunks() -> None:
    merged = merge_extractions(
        [
            extraction([character("Mr Weasley")]),
            extraction([character("Arthur Weasley", aliases=("Mr Weasley",))]),
            extraction([character("Percy Weasley", aliases=("Mr Weasley",))]),
        ]
    )

    assert len(merged.characters) == 3


def test_alias_only_overlap_stays_separate_even_without_possessives() -> None:
    merged = merge_extractions(
        [
            extraction(
                [
                    character("Arthur Weasley", aliases=("Mr Weasley",)),
                    character("Percy Weasley", aliases=("Mr Weasley",)),
                ]
            )
        ]
    )

    assert len(merged.characters) == 2


def test_name_matching_an_alias_merges_in_either_order() -> None:
    forward = merge_extractions(
        [extraction([character("Harry Potter", aliases=("The Boy Who Lived",)), character("The Boy Who Lived")])]
    )
    backward = merge_extractions(
        [extraction([character("The Boy Who Lived"), character("Harry Potter", aliases=("The Boy Who Lived",))])]
    )

    assert [c.name for c in forward.characters] == ["Harry Potter"]
    assert [c.name for c in backward.characters] == ["The Boy Who Lived"]


def test_relation_through_ambiguous_alias_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    merged = merge_extractions(
        [
            extraction(
                [
                    character("Tom Riddle", aliases=("Voldemort",)),
                    character("Lord Voldemort", aliases=("Voldemort",)),
                    character("Harry Potter"),
                ],
                [relation("Voldemort", "Harry Potter", "ENEMY_OF")],
            )
        ]
    )

    assert merged.relations == []
    assert caplog.text.count("ambiguous alias") == 1
    assert "unknown character" not in caplog.text


def test_relation_through_unambiguous_alias_is_attached() -> None:
    merged = merge_extractions(
        [
            extraction(
                [character("Tom Riddle", aliases=("Voldemort",)), character("Harry Potter")],
                [relation("Voldemort", "Harry Potter", "ENEMY_OF")],
            )
        ]
    )

    assert [(r.source, r.target) for r in merged.relations] == [("Tom Riddle", "Harry Potter")]


def test_possessive_descriptor_alias_is_dropped_but_name_is_kept(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    merged = merge_extractions(
        [extraction([character("Ronald", aliases=("Ron", "Ronald\u2019s father", "Ronald's mother"))])]
    )

    assert merged.characters[0].aliases == ["Ron"]
    assert "possessive" in caplog.text

    kept = merge_extractions([extraction([character("Ron's Dad"), character("O'Brien", aliases=("Mr. O'Brien",))])])
    assert [(c.name, c.aliases) for c in kept.characters] == [("Ron's Dad", []), ("O'Brien", ["Mr. O'Brien"])]


def test_relations_are_rewritten_to_canonical_names_and_deduplicated() -> None:
    merged = merge_extractions(
        [
            extraction(
                [character("Harry Potter", aliases=("Harry",)), character("Ron Weasley", aliases=("Ron",))],
                [relation("Harry", "Ron", "friend of", "Met on the train")],
            ),
            extraction(
                [character("Harry"), character("Ron")],
                [relation("Harry Potter", "Ron Weasley", "FRIEND_OF", "Share a dormitory")],
            ),
        ]
    )

    assert [(r.source, r.target, r.type, r.description) for r in merged.relations] == [
        ("Harry Potter", "Ron Weasley", "FRIEND_OF", "Met on the train")
    ]


def test_relations_to_unknown_characters_and_self_loops_are_dropped() -> None:
    merged = merge_extractions(
        [
            extraction(
                [character("Harry Potter", aliases=("Harry",))],
                [relation("Harry Potter", "Voldemort", "ENEMY_OF"), relation("Harry", "Harry Potter", "IS")],
            )
        ]
    )

    assert merged.relations == []


def _triples(merged) -> list[tuple[str, str, str, str]]:
    return [(r.source, r.type, r.target, r.description) for r in merged.relations]


@pytest.mark.parametrize(
    ("people", "extracted", "expected"),
    [
        (("Ron", "Arthur"), ("Ron", "Arthur", "CHILD_OF"), ("Arthur", "PARENT_OF", "Ron")),
        (("James", "Lord Voldemort"), ("James", "Lord Voldemort", "KILLED_BY"), ("Lord Voldemort", "KILLED", "James")),
        (("Harry", "Dudley"), ("Harry", "Dudley", "BULLIED_BY"), ("Dudley", "BULLIES", "Harry")),
        (("Harry", "Petunia"), ("Harry", "Petunia", "WARD_OF"), ("Petunia", "GUARDIAN_OF", "Harry")),
    ],
)
def test_inverse_relation_is_canonicalized(people, extracted, expected) -> None:
    merged = merge_extractions([extraction([character(n) for n in people], [relation(*extracted)])])

    assert [(r.source, r.type, r.target) for r in merged.relations] == [expected]


def test_inverse_and_canonical_forms_of_one_fact_collapse_keeping_the_first_description() -> None:
    merged = merge_extractions(
        [
            extraction(
                [character("Arthur"), character("Ron")],
                [
                    relation("Ron", "Arthur", "CHILD_OF", "Youngest son"),
                    relation("Arthur", "Ron", "PARENT_OF", "Father"),
                ],
            )
        ]
    )

    assert _triples(merged) == [("Arthur", "PARENT_OF", "Ron", "Youngest son")]


def test_other_types_keep_their_direction() -> None:
    merged = merge_extractions(
        [
            extraction(
                [character("Arthur"), character("Ron"), character("Harry")],
                [relation("Arthur", "Ron", "PARENT_OF"), relation("Ron", "Harry", "FRIEND_OF")],
            )
        ]
    )

    assert [(r.source, r.type, r.target) for r in merged.relations] == [
        ("Arthur", "PARENT_OF", "Ron"),
        ("Ron", "FRIEND_OF", "Harry"),
    ]

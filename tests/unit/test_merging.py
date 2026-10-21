import pytest

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


def test_alias_overlap_merges_transitively() -> None:
    merged = merge_extractions(
        [
            extraction([character("Tom Riddle", aliases=("Voldemort",))]),
            extraction([character("Lord Voldemort", aliases=("Voldemort", "You-Know-Who"))]),
        ]
    )

    assert len(merged.characters) == 1
    assert set(merged.characters[0].aliases) == {"Voldemort", "Lord Voldemort", "You-Know-Who"}


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

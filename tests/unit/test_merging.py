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


def test_relation_direction_is_preserved() -> None:
    merged = merge_extractions(
        [
            extraction(
                [character("Arthur Weasley"), character("Ron Weasley")],
                [
                    relation("Arthur Weasley", "Ron Weasley", "PARENT_OF"),
                    relation("Ron Weasley", "Arthur Weasley", "CHILD_OF"),
                ],
            )
        ]
    )

    assert {(r.source, r.type, r.target) for r in merged.relations} == {
        ("Arthur Weasley", "PARENT_OF", "Ron Weasley"),
        ("Ron Weasley", "CHILD_OF", "Arthur Weasley"),
    }

from app.llm.vocabulary import CANONICAL_RELATION_TYPES, INVERSE_RELATION_TYPES, SYMMETRIC_RELATION_TYPES


def test_canonical_relation_types_are_pinned() -> None:
    assert CANONICAL_RELATION_TYPES == (
        "PARENT_OF", "SIBLING_OF", "MARRIED_TO", "RELATIVE_OF",
        "GUARDIAN_OF",
        "FRIEND_OF", "ENEMY_OF", "ROMANTIC_PARTNER_OF",
        "WORKS_FOR", "TEACHER_OF", "MENTOR_OF",
        "BULLIES", "KILLED", "PROTECTS",
    )  # fmt: skip


def test_symmetric_types_are_canonical() -> None:
    assert SYMMETRIC_RELATION_TYPES.issubset(CANONICAL_RELATION_TYPES)


def test_inverses_map_onto_canonical_types_only() -> None:
    assert set(INVERSE_RELATION_TYPES.values()).issubset(CANONICAL_RELATION_TYPES)
    assert not set(INVERSE_RELATION_TYPES) & set(CANONICAL_RELATION_TYPES)


def test_symmetric_types_have_no_inverse() -> None:
    assert not SYMMETRIC_RELATION_TYPES & set(INVERSE_RELATION_TYPES)

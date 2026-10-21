CANONICAL_RELATION_TYPES: tuple[str, ...] = (
    "PARENT_OF", "SIBLING_OF", "MARRIED_TO", "RELATIVE_OF",
    "GUARDIAN_OF",
    "FRIEND_OF", "ENEMY_OF", "ROMANTIC_PARTNER_OF",
    "WORKS_FOR", "TEACHER_OF", "MENTOR_OF",
    "BULLIES", "KILLED", "PROTECTS",
)  # fmt: skip

SYMMETRIC_RELATION_TYPES = frozenset(
    {"SIBLING_OF", "MARRIED_TO", "RELATIVE_OF", "FRIEND_OF", "ENEMY_OF", "ROMANTIC_PARTNER_OF"}
)

# `X CHILD_OF Y` means `Y PARENT_OF X`. Extraction output is canonicalized with this map by swapping the ends,
# and queries use it to flip the direction. Stored data is never re-oriented.
INVERSE_RELATION_TYPES: dict[str, str] = {
    "CHILD_OF": "PARENT_OF",
    "WARD_OF": "GUARDIAN_OF",
    "EMPLOYS": "WORKS_FOR",
    "STUDENT_OF": "TEACHER_OF",
    "MENTEE_OF": "MENTOR_OF",
    "BULLIED_BY": "BULLIES",
    "KILLED_BY": "KILLED",
    "PROTECTED_BY": "PROTECTS",
}

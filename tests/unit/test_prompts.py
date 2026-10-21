import pytest

from app.llm.prompts import EXTRACTION_SYSTEM_PROMPT
from app.llm.vocabulary import CANONICAL_RELATION_TYPES, INVERSE_RELATION_TYPES


@pytest.mark.parametrize("relation_type", CANONICAL_RELATION_TYPES)
def test_extraction_prompt_lists_every_canonical_type(relation_type: str) -> None:
    assert relation_type in EXTRACTION_SYSTEM_PROMPT


def test_extraction_prompt_no_longer_asks_for_the_most_specific_type() -> None:
    assert "Use the most specific type" not in EXTRACTION_SYSTEM_PROMPT


def test_inverse_types_appear_only_in_the_sentence_that_forbids_them() -> None:
    assert "BULLIED_BY or CHILD_OF" in EXTRACTION_SYSTEM_PROMPT
    allowed = {"BULLIED_BY", "CHILD_OF"}
    assert not [key for key in INVERSE_RELATION_TYPES if key in EXTRACTION_SYSTEM_PROMPT and key not in allowed]

import pytest

from app.llm.prompts import EXTRACTION_SYSTEM_PROMPT
from app.llm.vocabulary import CANONICAL_RELATION_TYPES, INVERSE_RELATION_TYPES


@pytest.mark.parametrize("relation_type", CANONICAL_RELATION_TYPES)
def test_extraction_prompt_lists_every_canonical_type(relation_type: str) -> None:
    assert relation_type in EXTRACTION_SYSTEM_PROMPT


def test_extraction_prompt_no_longer_asks_for_the_most_specific_type() -> None:
    assert "Use the most specific type" not in EXTRACTION_SYSTEM_PROMPT


@pytest.mark.parametrize("inverse_type", INVERSE_RELATION_TYPES)
def test_extraction_prompt_lists_every_inverse_type(inverse_type: str) -> None:
    assert inverse_type in EXTRACTION_SYSTEM_PROMPT

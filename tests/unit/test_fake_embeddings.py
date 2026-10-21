"""Guards the fixtures of the integration tests against hash collisions of the 64-dimensional fake embedding."""

import math
from itertools import combinations

import pytest

from app.llm.vocabulary import CANONICAL_RELATION_TYPES
from app.pipeline.normalization import relation_type_phrase
from app.pipeline.resolution import character_embedding_text
from tests.factories import HERMIONE_DESCRIPTION, character
from tests.fakes import hashed_bag_of_words

DIM = 64
CANDIDATE_SCORE = 0.9  # relation_type_candidate_score of the integration settings

VOCAB = [relation_type_phrase(t) for t in CANONICAL_RELATION_TYPES]
# Phrases the integration tests use without an embedding override.
UNRELATED = ["arch nemesis of", "in love with", "pals with", "chums with", "rival of", "foe of", "buddy of"]


def score(left: str, right: str) -> float:
    a, b = hashed_bag_of_words(left, DIM), hashed_bag_of_words(right, DIM)
    return (1 + sum(x * y for x, y in zip(a, b, strict=True)) / (math.hypot(*a) * math.hypot(*b))) / 2


@pytest.mark.parametrize("phrase", UNRELATED)
def test_unrelated_phrase_is_not_a_candidate_of_any_canonical_type(phrase: str) -> None:
    assert [v for v in VOCAB if score(phrase, v) >= CANDIDATE_SCORE] == []


@pytest.mark.parametrize(("left", "right"), list(combinations(UNRELATED, 2)))
def test_unrelated_phrases_are_not_candidates_of_each_other(left: str, right: str) -> None:
    assert score(left, right) < CANDIDATE_SCORE


def test_lexical_fixture_ranks_hermione_above_james_by_vector() -> None:
    lily = character_embedding_text(character("Lily Potter", HERMIONE_DESCRIPTION))
    hermione = character_embedding_text(character("Hermione Granger", HERMIONE_DESCRIPTION))
    james = character_embedding_text(character("James Potter", "Head boy"))

    assert score(lily, hermione) > score(lily, james)

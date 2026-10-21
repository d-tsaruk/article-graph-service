import pytest

from app.llm.base import LLMError
from app.llm.judging import CharacterCard, RelationTypeCard
from tests.fakes import FakeLLMClient

KNOWN = [CharacterCard("S1", "Ron Weasley", ("Ron",), "Harry's friend")]


@pytest.fixture
def fake() -> FakeLLMClient:
    return FakeLLMClient(8)


async def test_declared_same_set_gives_stored(fake: FakeLLMClient) -> None:
    fake.same_characters = [{"Ron Weasley", "Ronald"}]

    judgement = await fake.judge_characters([CharacterCard("N1", "Ronald", (), "", ("S1",))], KNOWN)

    [verdict] = judgement.verdicts
    assert (verdict.item, verdict.decision, verdict.match) == ("N1", "stored", "S1")


async def test_two_new_cards_in_one_set_point_the_later_to_the_earlier(fake: FakeLLMClient) -> None:
    fake.same_characters = [{"Ron Weasley", "Ronald"}]
    new = [CharacterCard("N1", "Ron Weasley", (), ""), CharacterCard("N2", "Ronald", (), "")]

    judgement = await fake.judge_characters(new, [])

    assert [(v.decision, v.match) for v in judgement.verdicts] == [("new", None), ("same_as_new", "N1")]


async def test_without_a_same_set_the_verdict_is_new(fake: FakeLLMClient) -> None:
    judgement = await fake.judge_characters([CharacterCard("N1", "Ronald", (), "", ("S1",))], KNOWN)

    assert [(v.decision, v.match) for v in judgement.verdicts] == [("new", None)]


async def test_declared_type_pair_gives_stored(fake: FakeLLMClient) -> None:
    fake.same_relation_types = {("FRIENDS_WITH", "FRIEND_OF")}

    judgement = await fake.judge_relation_types(
        [RelationTypeCard("N1", "FRIENDS_WITH", ("S1",))], [RelationTypeCard("S1", "FRIEND_OF")]
    )

    [verdict] = judgement.verdicts
    assert (verdict.decision, verdict.match) == ("stored", "S1")


async def test_type_pair_between_new_cards_gives_same_as_new(fake: FakeLLMClient) -> None:
    fake.same_relation_types = {("CHUMS_WITH", "PALS_WITH")}
    new = [RelationTypeCard("N1", "PALS_WITH"), RelationTypeCard("N2", "CHUMS_WITH")]

    judgement = await fake.judge_relation_types(new, [])

    assert [(v.decision, v.match) for v in judgement.verdicts] == [("new", None), ("same_as_new", "N1")]


async def test_judge_error_is_raised_and_calls_are_recorded(fake: FakeLLMClient) -> None:
    fake.judge_error = LLMError("boom")

    with pytest.raises(LLMError, match="boom"):
        await fake.judge_characters([CharacterCard("N1", "A", (), "")], [])
    with pytest.raises(LLMError, match="boom"):
        await fake.judge_relation_types([RelationTypeCard("N1", "A_OF")], [])

    assert len(fake.character_judge_calls) == 1
    assert len(fake.relation_type_judge_calls) == 1

import logging

import pytest

from app.llm.schemas import Judgement, Verdict
from app.pipeline.adjudication import Group, group_verdicts


def verdict(item: str, decision: str, match: str | None = None) -> Verdict:
    return Verdict(item=item, reason="r", decision=decision, match=match)  # type: ignore[arg-type]


def judgement(*verdicts: Verdict) -> Judgement:
    return Judgement(verdicts=list(verdicts))


CANDIDATES = {"N1": ("S1", "S2"), "N2": ("S1",), "N3": ()}


def singletons(refs: tuple[str, ...]) -> list[Group]:
    return [Group(members=(ref,), canonical=ref, stored=None) for ref in refs]


def test_no_judgement_gives_singletons() -> None:
    assert group_verdicts(CANDIDATES, None) == singletons(("N1", "N2", "N3"))


def test_valid_stored_verdict() -> None:
    result = group_verdicts(CANDIDATES, judgement(verdict("N1", "stored", "S2")))

    assert result[0] == Group(members=("N1",), canonical="N1", stored="S2")


def test_stored_match_outside_the_candidates_becomes_new(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="app.pipeline.adjudication"):
        result = group_verdicts(CANDIDATES, judgement(verdict("N2", "stored", "S2")))

    assert result[1] == Group(members=("N2",), canonical="N2", stored=None)
    assert "N2" in caplog.text


def test_stored_without_a_match_becomes_new() -> None:
    result = group_verdicts(CANDIDATES, judgement(verdict("N1", "stored", None)))

    assert result == singletons(("N1", "N2", "N3"))


@pytest.mark.parametrize("match", ["N1", "N9", None])
def test_invalid_same_as_new_becomes_new(match: str | None) -> None:
    result = group_verdicts(CANDIDATES, judgement(verdict("N1", "same_as_new", match)))

    assert result == singletons(("N1", "N2", "N3"))


def test_chain_merges_into_one_group_with_the_stored_root() -> None:
    candidates = {"N1": ("S1",), "N2": (), "N3": ()}
    result = group_verdicts(
        candidates,
        judgement(
            verdict("N1", "stored", "S1"), verdict("N2", "same_as_new", "N1"), verdict("N3", "same_as_new", "N2")
        ),
    )

    assert result == [Group(members=("N1", "N2", "N3"), canonical="N1", stored="S1")]


def test_cycle_keeps_the_first_member_as_canonical_and_stores_nothing() -> None:
    result = group_verdicts(
        {"N1": (), "N2": ()}, judgement(verdict("N1", "same_as_new", "N2"), verdict("N2", "same_as_new", "N1"))
    )

    assert result == [Group(members=("N1", "N2"), canonical="N1", stored=None)]


def test_root_is_canonical_even_when_it_comes_later() -> None:
    result = group_verdicts({"N1": (), "N2": ()}, judgement(verdict("N1", "same_as_new", "N2")))

    assert result == [Group(members=("N1", "N2"), canonical="N2", stored=None)]


def test_missing_verdict_counts_as_new(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="app.pipeline.adjudication"):
        result = group_verdicts(CANDIDATES, judgement(verdict("N1", "stored", "S1")))

    assert result[1:] == singletons(("N2", "N3"))
    assert "N2" in caplog.text


def test_verdict_for_an_unknown_item_is_ignored() -> None:
    result = group_verdicts({"N1": ("S1",)}, judgement(verdict("N7", "stored", "S1"), verdict("N1", "new")))

    assert result == singletons(("N1",))


def test_first_verdict_wins_over_duplicates() -> None:
    result = group_verdicts({"N1": ("S1",)}, judgement(verdict("N1", "stored", "S1"), verdict("N1", "new")))

    assert result == [Group(members=("N1",), canonical="N1", stored="S1")]


def test_two_stored_items_never_merge() -> None:
    result = group_verdicts(
        {"N1": ("S1",), "N2": ("S2",)}, judgement(verdict("N1", "stored", "S1"), verdict("N2", "stored", "S2"))
    )

    assert [group.stored for group in result] == ["S1", "S2"]


def test_warnings_name_the_judge_the_items_and_the_reason(caplog: pytest.LogCaptureFixture) -> None:
    labels = {"N1": "Ronald", "N2": "Percy", "S1": "Ron Weasley"}
    bad = Judgement(verdicts=[Verdict(item="N1", reason="looks alike", decision="stored", match="S9")])

    with caplog.at_level(logging.WARNING, logger="app.pipeline.adjudication"):
        group_verdicts({"N1": ("S1",), "N2": ()}, bad, kind="character", labels=labels)

    assert "character judge" in caplog.text
    assert "Ronald" in caplog.text
    assert "looks alike" in caplog.text
    assert "Percy" in caplog.text  # the item without a verdict

import neo4j
import pytest
from fastapi.testclient import TestClient

from app.llm.vocabulary import CANONICAL_RELATION_TYPES
from tests.factories import HARRY_DESCRIPTION, RON_DESCRIPTION, character, extraction, relation
from tests.fakes import FakeLLMClient

HARRY = character("Harry Potter", HARRY_DESCRIPTION)
RON = character("Ron Weasley", RON_DESCRIPTION)


def upload_markdown(client: TestClient, headers: dict[str, str], text: str = "Harry and Ron.") -> dict:
    response = client.post("/articles/upload", data={"text": text}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


def edge_types(db: neo4j.Driver) -> set[str]:
    records, _, _ = db.execute_query("MATCH ()-[r:RELATED_TO]->() RETURN r.type AS type")
    return {r["type"] for r in records}


def type_nodes(db: neo4j.Driver) -> set[str]:
    records, _, _ = db.execute_query("MATCH (t:RelationType) RETURN t.name AS name")
    return {r["name"] for r in records}


def offered(fake_llm: FakeLLMClient, call: int = 0) -> dict[str, tuple[str, ...]]:
    """New type name -> names of the known types offered as its candidates."""
    new, known = fake_llm.relation_type_judge_calls[call]
    names = {card.ref: card.name for card in known}
    return {card.name: tuple(names[ref] for ref in card.candidates) for card in new}


@pytest.mark.parametrize(
    ("new", "target", "override"),
    [
        ("FRIENDS_WITH", "FRIEND_OF", {"friends with": "friend of"}),
        ("SPOUSE_OF", "MARRIED_TO", {"spouse of": "married to"}),
        ("FATHER_OF", "PARENT_OF", {"father of": "parent of"}),
        ("BEST_FRIENDS_WITH", "FRIEND_OF", {"best friends with": "friend of"}),
    ],
)
def test_specific_or_synonym_type_maps_to_canonical(
    client: TestClient,
    auth_headers: dict[str, str],
    fake_llm: FakeLLMClient,
    db: neo4j.Driver,
    new: str,
    target: str,
    override: dict[str, str],
) -> None:
    fake_llm.embedding_overrides = override
    fake_llm.same_relation_types = {(new, target)}
    fake_llm.extraction = extraction([HARRY, RON], [relation("Harry Potter", "Ron Weasley", new)])

    upload_markdown(client, auth_headers)

    assert edge_types(db) == {target}
    assert new not in type_nodes(db)
    assert target in offered(fake_llm)[new]


@pytest.mark.parametrize(
    ("new", "target", "override", "is_offered"),
    [
        ("BULLIED_BY", "BULLIES", {"bullied by": "bullies"}, False),  # a known inverse is filtered in code
        ("CHILD_OF", "PARENT_OF", {"child of": "parent of"}, False),
        ("DIVORCED_FROM", "MARRIED_TO", {"divorced from": "married to"}, True),  # an opposite reaches the judge
    ],
)
def test_inverse_or_opposite_type_is_not_merged(
    client: TestClient,
    auth_headers: dict[str, str],
    fake_llm: FakeLLMClient,
    db: neo4j.Driver,
    new: str,
    target: str,
    override: dict[str, str],
    is_offered: bool,
) -> None:
    fake_llm.embedding_overrides = override
    fake_llm.extraction = extraction([HARRY, RON], [relation("Harry Potter", "Ron Weasley", new)])

    upload_markdown(client, auth_headers)

    if is_offered:
        assert target in offered(fake_llm)[new]
    elif fake_llm.relation_type_judge_calls:  # a colliding fake embedding may still offer other types
        assert target not in offered(fake_llm)[new]
    assert edge_types(db) == {new}
    assert new in type_nodes(db)


def test_inverse_type_never_merges_even_when_the_judge_says_stored(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.embedding_overrides = {"bullied by": "bullies"}
    fake_llm.same_relation_types = {("BULLIED_BY", "BULLIES")}
    fake_llm.extraction = extraction([HARRY, RON], [relation("Harry Potter", "Ron Weasley", "BULLIED_BY")])

    upload_markdown(client, auth_headers)

    assert edge_types(db) == {"BULLIED_BY"}
    assert fake_llm.relation_type_judge_calls == []  # the inverse was never offered


def test_inverse_type_is_not_offered_but_other_candidates_are(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    fake_llm.embedding_overrides = {"bullied by": "bullies", "pals with": "friend of"}
    fake_llm.extraction = extraction(
        [HARRY, RON],
        [relation("Harry Potter", "Ron Weasley", "BULLIED_BY"), relation("Ron Weasley", "Harry Potter", "PALS_WITH")],
    )

    upload_markdown(client, auth_headers)

    candidates = offered(fake_llm)
    assert "BULLIES" not in candidates["BULLIED_BY"]
    assert "FRIEND_OF" in candidates["PALS_WITH"]


def test_synonym_types_in_one_upload_use_one_judge_call(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.same_relation_types = {("CHUMS_WITH", "PALS_WITH")}
    fake_llm.extraction = extraction(
        [HARRY, RON],
        [relation("Harry Potter", "Ron Weasley", "PALS_WITH"), relation("Ron Weasley", "Harry Potter", "CHUMS_WITH")],
    )

    upload_markdown(client, auth_headers)

    assert len(fake_llm.relation_type_judge_calls) == 1
    assert edge_types(db) == {"PALS_WITH"}
    assert "CHUMS_WITH" not in type_nodes(db)


def test_two_known_synonyms_in_one_upload(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.embedding_overrides = {"friends with": "friend of", "spouse of": "married to"}
    fake_llm.same_relation_types = {("FRIENDS_WITH", "FRIEND_OF"), ("SPOUSE_OF", "MARRIED_TO")}
    fake_llm.extraction = extraction(
        [HARRY, RON],
        [
            relation("Harry Potter", "Ron Weasley", "FRIENDS_WITH"),
            relation("Ron Weasley", "Harry Potter", "SPOUSE_OF"),
        ],
    )

    upload_markdown(client, auth_headers)

    assert len(fake_llm.relation_type_judge_calls) == 1
    assert edge_types(db) == {"FRIEND_OF", "MARRIED_TO"}


def test_unrelated_type_is_not_offered(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    fake_llm.embedding_overrides = {"friends with": "friend of"}
    fake_llm.extraction = extraction(
        [HARRY, RON],
        [
            relation("Harry Potter", "Ron Weasley", "FRIENDS_WITH"),
            relation("Ron Weasley", "Harry Potter", "ARCH_NEMESIS_OF"),
        ],
    )

    upload_markdown(client, auth_headers)

    assert len(fake_llm.relation_type_judge_calls) == 1
    new, _ = fake_llm.relation_type_judge_calls[0]
    assert next(card for card in new if card.name == "ARCH_NEMESIS_OF").candidates == ()


def test_single_new_type_without_candidates_makes_no_judge_call(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.extraction = extraction([HARRY, RON], [relation("Harry Potter", "Ron Weasley", "ARCH_NEMESIS_OF")])

    upload_markdown(client, auth_headers)

    assert fake_llm.relation_type_judge_calls == []
    assert "ARCH_NEMESIS_OF" in type_nodes(db)


def test_vocabulary_is_seeded_on_first_upload(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.extraction = extraction([HARRY, RON], [relation("Harry Potter", "Ron Weasley", "FRIEND_OF")])

    upload_markdown(client, auth_headers)

    assert type_nodes(db) >= set(CANONICAL_RELATION_TYPES)
    records, _, _ = db.execute_query("MATCH (t:RelationType) RETURN t.embedding AS embedding")
    assert all(len(r["embedding"]) == fake_llm.embedding_dimensions for r in records)


def test_type_used_only_by_a_dropped_self_loop_gets_no_node(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.same_characters = [{"Ron Weasley", "Ronald"}]
    fake_llm.extraction = extraction(
        [RON, character("Ronald", RON_DESCRIPTION)], [relation("Ron Weasley", "Ronald", "RIVAL_OF")]
    )

    upload_markdown(client, auth_headers)

    assert "RIVAL_OF" not in type_nodes(db)
    assert edge_types(db) == set()

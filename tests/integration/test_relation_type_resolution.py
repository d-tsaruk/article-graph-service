from collections.abc import AsyncIterator

import neo4j
import pytest
from fastapi.testclient import TestClient
from neo4j import AsyncGraphDatabase

from app.core.config import Settings
from app.graph.repository import GraphRepository
from app.llm.vocabulary import CANONICAL_RELATION_TYPES
from app.pipeline.resolution import RelationTypeResolver
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


def edges(db: neo4j.Driver) -> set[tuple[str, str, str]]:
    records, _, _ = db.execute_query(
        "MATCH (s:Character)-[r:RELATED_TO]->(t:Character) RETURN s.name AS s, r.type AS type, t.name AS t"
    )
    return {(r["s"], r["type"], r["t"]) for r in records}


def test_opposite_type_reaches_the_judge_but_is_not_merged(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.embedding_overrides = {"divorced from": "married to"}
    fake_llm.extraction = extraction([HARRY, RON], [relation("Harry Potter", "Ron Weasley", "DIVORCED_FROM")])

    upload_markdown(client, auth_headers)

    assert "MARRIED_TO" in offered(fake_llm)["DIVORCED_FROM"]
    assert edge_types(db) == {"DIVORCED_FROM"}
    assert "DIVORCED_FROM" in type_nodes(db)


@pytest.mark.parametrize(
    ("new", "stored", "override", "expected"),
    [
        ("BULLIED_BY", "BULLIES", {"bullied by": "bullies"}, ("Ron Weasley", "BULLIES", "Harry Potter")),
        ("CHILD_OF", "PARENT_OF", {"child of": "parent of"}, ("Ron Weasley", "PARENT_OF", "Harry Potter")),
    ],
)
def test_inverse_type_is_canonicalized_before_the_judge(
    client: TestClient,
    auth_headers: dict[str, str],
    fake_llm: FakeLLMClient,
    db: neo4j.Driver,
    new: str,
    stored: str,
    override: dict[str, str],
    expected: tuple[str, str, str],
) -> None:
    fake_llm.embedding_overrides = override
    fake_llm.same_relation_types = {(new, stored)}  # inert: the canonical type is stored, so the judge is never asked
    fake_llm.extraction = extraction([HARRY, RON], [relation("Harry Potter", "Ron Weasley", new)])

    upload_markdown(client, auth_headers)

    assert edges(db) == {expected}
    assert new not in type_nodes(db)
    assert fake_llm.relation_type_judge_calls == []


def test_inverse_types_are_canonicalized_end_to_end(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.extraction = extraction(
        [character("Ron Weasley", RON_DESCRIPTION), character("Arthur Weasley"), character("James Potter"),
         character("Lord Voldemort", aliases=("Voldemort",))],
        [
            relation("Ron Weasley", "Arthur Weasley", "CHILD_OF"),
            relation("James Potter", "Lord Voldemort", "KILLED_BY"),
        ],
    )  # fmt: skip

    upload_markdown(client, auth_headers)

    assert edges(db) == {
        ("Arthur Weasley", "PARENT_OF", "Ron Weasley"),
        ("Lord Voldemort", "KILLED", "James Potter"),
    }
    assert not {"CHILD_OF", "KILLED_BY"} & (edge_types(db) | type_nodes(db))
    assert fake_llm.relation_type_judge_calls == []


@pytest.fixture
async def resolver_with_repository(
    settings: Settings, fake_llm: FakeLLMClient, db: neo4j.Driver
) -> AsyncIterator[RelationTypeResolver]:
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value())
    )
    try:
        repository = GraphRepository(driver, settings.neo4j_database, settings.embedding_dimensions)
        await repository.init_schema()
        yield RelationTypeResolver(repository, fake_llm, candidate_score=settings.relation_type_candidate_score)
    finally:
        await driver.close()


async def test_resolver_never_merges_a_direct_inverse_even_when_the_judge_says_stored(
    resolver_with_repository: RelationTypeResolver, fake_llm: FakeLLMClient
) -> None:
    fake_llm.embedding_overrides = {"bullied by": "bullies"}
    fake_llm.same_relation_types = {("BULLIED_BY", "BULLIES")}

    plan = await resolver_with_repository.plan(["BULLIED_BY"])

    assert plan.mapping == {"BULLIED_BY": "BULLIED_BY"}
    assert fake_llm.relation_type_judge_calls == []  # the inverse was never offered


async def test_resolver_does_not_offer_an_inverse_but_offers_other_candidates(
    resolver_with_repository: RelationTypeResolver, fake_llm: FakeLLMClient
) -> None:
    fake_llm.embedding_overrides = {"bullied by": "bullies", "pals with": "friend of"}

    await resolver_with_repository.plan(["BULLIED_BY", "PALS_WITH"])

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

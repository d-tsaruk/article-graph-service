import neo4j
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.llm.base import LLMError
from app.llm.schemas import ExtractedCharacter, ExtractedRelation
from app.main import create_app
from tests.factories import (
    HARRY_DESCRIPTION,
    HERMIONE_DESCRIPTION,
    RON_DESCRIPTION,
    character,
    extraction,
    relation,
)
from tests.fakes import FakeLLMClient
from tests.integration.conftest import APP_TOKEN


def upload_markdown(client: TestClient, headers: dict[str, str], text: str) -> dict:
    response = client.post("/articles/upload", data={"text": text}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


def upload(
    client: TestClient,
    headers: dict[str, str],
    fake: FakeLLMClient,
    characters: list[ExtractedCharacter],
    relations: list[ExtractedRelation] | None = None,
    *,
    text: str,
) -> dict:
    fake.extraction = extraction(characters, relations)
    return upload_markdown(client, headers, text)


def count(db: neo4j.Driver, label: str) -> int:
    records, _, _ = db.execute_query(f"MATCH (n:{label}) RETURN count(n) AS n")
    return records[0]["n"]


def offered_names(fake: FakeLLMClient, new_name: str) -> list[str]:
    """Names of the stored characters offered to `new_name` in the last judge call, in offered order."""
    new, known = fake.character_judge_calls[-1]
    names = {card.ref: card.name for card in known}
    card = next(card for card in new if card.name == new_name)
    return [names[ref] for ref in card.candidates]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            character("Arthur Weasley", "Ministry worker, father of Ron"),
            character("Molly Weasley", "Mother of Ron, runs the Burrow"),
        ),
        (character("Vernon Dursley", "Harry's uncle"), character("Petunia Dursley", "Harry's aunt")),
        (character("Ron Weasley", RON_DESCRIPTION), character("Percy Weasley", "Ron's older brother, a prefect")),
    ],
)
def test_family_members_sharing_a_surname_stay_separate(
    client: TestClient,
    auth_headers: dict[str, str],
    fake_llm: FakeLLMClient,
    db: neo4j.Driver,
    first: ExtractedCharacter,
    second: ExtractedCharacter,
) -> None:
    fake_llm.same_characters = [{first.name, second.name}]  # a judge that wrongly merges them
    upload(client, auth_headers, fake_llm, [first], text="first")
    upload(client, auth_headers, fake_llm, [second], text="second")

    assert fake_llm.character_judge_calls == []  # the stored sibling is never even offered
    assert count(db, "Character") == 2


def test_new_characters_with_conflicting_names_are_not_merged_by_the_judge(
    client: TestClient,
    auth_headers: dict[str, str],
    fake_llm: FakeLLMClient,
    db: neo4j.Driver,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_llm.same_characters = [{"Ron Weasley", "Percy Weasley"}]

    upload(
        client,
        auth_headers,
        fake_llm,
        [character("Ron Weasley", RON_DESCRIPTION), character("Percy Weasley", "Ron's older brother, a prefect")],
        text="one article",
    )

    records, _, _ = db.execute_query("MATCH (c:Character) RETURN c.name AS name ORDER BY name")
    assert [r["name"] for r in records] == ["Percy Weasley", "Ron Weasley"]
    assert "conflicting names" in caplog.text


def test_given_name_prefix_is_not_a_conflict(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.same_characters = [{"Ronald Weasley", "Ron Weasley"}]
    upload(client, auth_headers, fake_llm, [character("Ronald Weasley", RON_DESCRIPTION)], text="first")

    upload(client, auth_headers, fake_llm, [character("Ron Weasley", "Harry's best friend")], text="second")

    assert offered_names(fake_llm, "Ron Weasley") == ["Ronald Weasley"]
    records, _, _ = db.execute_query("MATCH (c:Character) RETURN c.name AS name, c.aliases AS aliases")
    assert [(r["name"], r["aliases"]) for r in records] == [("Ronald Weasley", ["Ron Weasley"])]


@pytest.mark.parametrize(
    ("canonical", "alias"),
    [("Ron Weasley", "Ronald"), ("Lord Voldemort", "Tom Riddle"), ("Harry Potter", "The Boy Who Lived")],
)
def test_aliases_merge_when_judge_confirms(
    client: TestClient,
    auth_headers: dict[str, str],
    fake_llm: FakeLLMClient,
    db: neo4j.Driver,
    canonical: str,
    alias: str,
) -> None:
    fake_llm.same_characters = [{canonical, alias}]
    upload(client, auth_headers, fake_llm, [character(canonical, "A wizard")], text="first")
    upload(client, auth_headers, fake_llm, [character(alias, "A wizard")], text="second")

    records, _, _ = db.execute_query(
        "MATCH (c:Character) OPTIONAL MATCH (:Article)-[m:MENTIONS]->(c) "
        "RETURN c.name AS name, c.aliases AS aliases, count(m) AS mentions"
    )
    assert [(r["name"], r["mentions"]) for r in records] == [(canonical, 2)]
    assert alias in records[0]["aliases"]


def test_in_article_duplicates_merge(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.same_characters = [{"Ron Weasley", "Ronald"}]

    upload(
        client,
        auth_headers,
        fake_llm,
        [character("Ron Weasley", RON_DESCRIPTION), character("Ronald", "Harry's best friend")],
        text="one article",
    )

    records, _, _ = db.execute_query("MATCH (c:Character) RETURN c.name AS name, c.aliases AS aliases")
    assert [(r["name"], r["aliases"]) for r in records] == [("Ron Weasley", ["Ronald"])]


def test_in_article_alias_only_overlap_is_decided_by_the_judge(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.same_characters = [{"Tom Riddle", "Lord Voldemort"}]

    upload(
        client,
        auth_headers,
        fake_llm,
        [
            character("Tom Riddle", "A Hogwarts student", aliases=("Voldemort",)),
            character("Lord Voldemort", "A dark wizard", aliases=("Voldemort", "You-Know-Who")),
        ],
        text="one article",
    )

    new, _ = fake_llm.character_judge_calls[-1]
    assert {card.name: set(card.aliases) for card in new} == {
        "Tom Riddle": {"Voldemort"},
        "Lord Voldemort": {"Voldemort", "You-Know-Who"},
    }
    records, _, _ = db.execute_query("MATCH (c:Character) RETURN c.name AS name, c.aliases AS aliases")
    assert [(r["name"], set(r["aliases"])) for r in records] == [
        ("Tom Riddle", {"Voldemort", "Lord Voldemort", "You-Know-Who"})
    ]


def test_alias_collision_goes_to_judge(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    upload(
        client,
        auth_headers,
        fake_llm,
        [character("Ron Weasley", RON_DESCRIPTION, aliases=("Ron", "Weasley"))],
        text="first",
    )
    upload(
        client,
        auth_headers,
        fake_llm,
        [character("Mr. Weasley", "Ron's father", aliases=("Weasley",))],
        text="second",
    )

    assert "Ron Weasley" in offered_names(fake_llm, "Mr. Weasley")
    assert count(db, "Character") == 2


def test_lexical_candidate_is_offered_beyond_vector_top_k(
    settings: Settings, fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    limited = settings.model_copy(update={"character_candidate_limit": 1})
    with TestClient(create_app(limited, llm_client=fake_llm)) as client:
        token = client.post("/auth/token", json={"app_token": APP_TOKEN}).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        upload(
            client,
            headers,
            fake_llm,
            [character("James Potter", "Head boy"), character("Hermione Granger", HERMIONE_DESCRIPTION)],
            text="stored",
        )

        upload(client, headers, fake_llm, [character("Mrs. Potter", HERMIONE_DESCRIPTION)], text="new")

    # Lexical channel first (shares "potter"), then the single vector neighbour.
    assert offered_names(fake_llm, "Mrs. Potter") == ["James Potter", "Hermione Granger"]


def test_exact_canonical_name_skips_judge(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    upload(client, auth_headers, fake_llm, [character("Harry Potter", HARRY_DESCRIPTION)], text="first")

    upload(client, auth_headers, fake_llm, [character("Harry Potter", "Seeker")], text="second")

    assert fake_llm.character_judge_calls == []
    records, _, _ = db.execute_query(
        "MATCH (c:Character) OPTIONAL MATCH (:Article)-[m:MENTIONS]->(c) RETURN count(DISTINCT c) AS c, count(m) AS m"
    )
    assert (records[0]["c"], records[0]["m"]) == (1, 2)


def test_single_new_character_on_empty_graph_makes_no_judge_call(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    upload(client, auth_headers, fake_llm, [character("Harry Potter", HARRY_DESCRIPTION)], text="only one")

    assert fake_llm.character_judge_calls == []


def test_one_character_judge_call_per_upload(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    upload(client, auth_headers, fake_llm, [character("Harry Potter", HARRY_DESCRIPTION)], text="first")

    upload(
        client,
        auth_headers,
        fake_llm,
        [
            character("Ron Weasley", RON_DESCRIPTION),
            character("Hermione Granger", HERMIONE_DESCRIPTION),
            character("Draco Malfoy", "Slytherin rival"),
        ],
        text="second",
    )

    assert len(fake_llm.character_judge_calls) == 1
    new, _ = fake_llm.character_judge_calls[0]
    assert len(new) == 3


def test_judge_failure_leaves_graph_unchanged(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.embedding_overrides = {"friends with": "friend of"}
    upload(
        client,
        auth_headers,
        fake_llm,
        [character("Harry Potter", HARRY_DESCRIPTION, aliases=("Harry",)), character("Ron Weasley", RON_DESCRIPTION)],
        [relation("Harry Potter", "Ron Weasley", "FRIEND_OF")],
        text="first",
    )

    def snapshot() -> tuple[int, int, int, list[str]]:
        records, _, _ = db.execute_query("MATCH (c:Character {name: 'Harry Potter'}) RETURN c.aliases AS aliases")
        return count(db, "Article"), count(db, "Character"), count(db, "RelationType"), records[0]["aliases"]

    before = snapshot()
    fake_llm.judge_error = LLMError("boom")
    fake_llm.extraction = extraction(
        [character("Ronald", RON_DESCRIPTION), character("Hermione Granger", HERMIONE_DESCRIPTION)],
        [relation("Ronald", "Hermione Granger", "FRIENDS_WITH")],
    )

    response = client.post("/articles/upload", data={"text": "second"}, headers=auth_headers)

    assert response.status_code == 502
    assert snapshot() == before


def test_character_judge_failure_alone_leaves_graph_unchanged(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    upload(
        client,
        auth_headers,
        fake_llm,
        [character("Harry Potter", HARRY_DESCRIPTION), character("Ron Weasley", RON_DESCRIPTION)],
        [relation("Harry Potter", "Ron Weasley", "FRIEND_OF")],
        text="first",
    )
    before = (count(db, "Article"), count(db, "Character"), count(db, "RelationType"))
    judge_calls = len(fake_llm.character_judge_calls)
    fake_llm.judge_error = LLMError("boom")
    fake_llm.extraction = extraction(
        [character("Ronald", RON_DESCRIPTION), character("Hermione Granger", HERMIONE_DESCRIPTION)],
        [relation("Ronald", "Hermione Granger", "FRIEND_OF")],  # known type: only the character judge runs
    )

    response = client.post("/articles/upload", data={"text": "second"}, headers=auth_headers)

    assert response.status_code == 502
    assert fake_llm.relation_type_judge_calls == []
    assert len(fake_llm.character_judge_calls) == judge_calls + 1
    assert (count(db, "Article"), count(db, "Character"), count(db, "RelationType")) == before

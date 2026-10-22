import neo4j
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.llm.base import UnconfiguredLLMClient
from app.main import create_app
from tests.factories import (
    HARRY_DESCRIPTION,
    HERMIONE_DESCRIPTION,
    RON_DESCRIPTION,
    character,
    extraction,
    insert_character,
    relation,
)
from tests.fakes import FakeLLMClient
from tests.integration.conftest import APP_TOKEN


def seed(client: TestClient, headers: dict[str, str], fake_llm: FakeLLMClient) -> dict[str, dict]:
    fake_llm.extraction = extraction(
        [
            character("Harry Potter", HARRY_DESCRIPTION),
            character("Ron Weasley", RON_DESCRIPTION),
            character("Hermione Granger", HERMIONE_DESCRIPTION),
        ],
        [
            relation("Harry Potter", "Ron Weasley", "FRIEND_OF"),
            relation("Hermione Granger", "Harry Potter", "FRIEND_OF"),
            relation("Ron Weasley", "Hermione Granger", "IN_LOVE_WITH"),
        ],
    )
    response = client.post("/articles/upload", data={"text": "# Book\n\nStory."}, headers=headers)
    assert response.status_code == 201, response.text
    body = response.json()
    return {"article": body, **{c["name"]: c for c in body["characters"]}}


def test_list_characters(client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient) -> None:
    seed(client, auth_headers, fake_llm)

    response = client.get("/characters", headers=auth_headers)

    assert response.status_code == 200
    assert [c["name"] for c in response.json()] == ["Harry Potter", "Hermione Granger", "Ron Weasley"]
    assert "embedding" not in response.json()[0]


def test_list_characters_pagination(client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient) -> None:
    seed(client, auth_headers, fake_llm)

    response = client.get("/characters?limit=1&offset=1", headers=auth_headers)

    assert [c["name"] for c in response.json()] == ["Hermione Granger"]


def test_characters_by_article(client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient) -> None:
    seeded = seed(client, auth_headers, fake_llm)
    fake_llm.extraction = extraction([character("Draco Malfoy", "Slytherin rival")])
    other = client.post("/articles/upload", data={"text": "Draco."}, headers=auth_headers).json()

    first = client.get(f"/characters/by_article/{seeded['article']['article_id']}", headers=auth_headers)
    second = client.get(f"/characters/by_article/{other['article_id']}", headers=auth_headers)

    assert {c["name"] for c in first.json()} == {"Harry Potter", "Ron Weasley", "Hermione Granger"}
    assert [c["name"] for c in second.json()] == ["Draco Malfoy"]


def test_characters_by_unknown_article_returns_404(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.get("/characters/by_article/missing", headers=auth_headers)

    assert response.status_code == 404


def test_relation_types(client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient) -> None:
    seed(client, auth_headers, fake_llm)

    response = client.get("/relations/types", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == [{"type": "FRIEND_OF", "count": 2}, {"type": "IN_LOVE_WITH", "count": 1}]


def test_connected_characters_in_any_direction(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    seeded = seed(client, auth_headers, fake_llm)
    harry_id = seeded["Harry Potter"]["id"]

    response = client.get(f"/relations/connected/{harry_id}?type=FRIEND_OF", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["resolved_type"] == "FRIEND_OF"
    assert {(c["character"]["name"], c["direction"]) for c in body["connections"]} == {
        ("Ron Weasley", "outgoing"),
        ("Hermione Granger", "outgoing"),
    }


def test_connected_characters_filtered_by_direction(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    seeded = seed(client, auth_headers, fake_llm)
    harry_id = seeded["Harry Potter"]["id"]

    response = client.get(f"/relations/connected/{harry_id}?type=FRIEND_OF&direction=incoming", headers=auth_headers)

    assert {c["character"]["name"] for c in response.json()["connections"]} == {"Ron Weasley", "Hermione Granger"}


def test_connected_characters_normalizes_requested_type(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    seeded = seed(client, auth_headers, fake_llm)
    ron_id = seeded["Ron Weasley"]["id"]
    judge_calls = len(fake_llm.relation_type_judge_calls)

    response = client.get(f"/relations/connected/{ron_id}?type=in love with", headers=auth_headers)

    assert len(fake_llm.relation_type_judge_calls) == judge_calls

    body = response.json()
    assert body["resolved_type"] == "IN_LOVE_WITH"
    assert [c["character"]["name"] for c in body["connections"]] == ["Hermione Granger"]


def test_connected_characters_with_unknown_type_returns_empty_list(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    seeded = seed(client, auth_headers, fake_llm)

    response = client.get(
        f"/relations/connected/{seeded['Harry Potter']['id']}?type=ARCH_NEMESIS_OF", headers=auth_headers
    )

    assert response.status_code == 200
    assert response.json()["resolved_type"] is None
    assert response.json()["connections"] == []


def test_connected_characters_for_unknown_character_returns_404(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    response = client.get("/relations/connected/missing?type=FRIEND_OF", headers=auth_headers)
    unknown_type = client.get("/relations/connected/missing?type=buddy of", headers=auth_headers)

    assert response.status_code == 404
    assert unknown_type.status_code == 404
    assert fake_llm.embed_calls == []
    assert fake_llm.relation_type_judge_calls == []


def test_connected_characters_inverse_type_uses_canonical_type(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    fake_llm.extraction = extraction(
        [character("Lily Potter", "Harry's mother"), character("Harry Potter", HARRY_DESCRIPTION)],
        [relation("Lily Potter", "Harry Potter", "PARENT_OF")],
    )
    body = client.post("/articles/upload", data={"text": "Lily and Harry."}, headers=auth_headers).json()
    ids = {c["name"]: c["id"] for c in body["characters"]}

    response = client.get(f"/relations/connected/{ids['Harry Potter']}?type=child of", headers=auth_headers)
    outgoing = client.get(
        f"/relations/connected/{ids['Harry Potter']}?type=child of&direction=outgoing", headers=auth_headers
    )
    from_lily = client.get(
        f"/relations/connected/{ids['Lily Potter']}?type=child of&direction=outgoing", headers=auth_headers
    )

    assert response.json()["resolved_type"] == "PARENT_OF"
    assert [c["character"]["name"] for c in response.json()["connections"]] == ["Lily Potter"]
    assert [c["character"]["name"] for c in outgoing.json()["connections"]] == ["Lily Potter"]
    assert from_lily.json()["connections"] == []
    assert fake_llm.relation_type_judge_calls == []


def test_inverse_map_wins_over_a_stored_node_of_the_inverse_type(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.extraction = extraction(
        [character("Lily Potter", "Harry's mother"), character("Harry Potter", HARRY_DESCRIPTION)],
        [relation("Lily Potter", "Harry Potter", "PARENT_OF")],
    )
    body = client.post("/articles/upload", data={"text": "Lily and Harry."}, headers=auth_headers).json()
    harry = next(c["id"] for c in body["characters"] if c["name"] == "Harry Potter")
    db.execute_query("CREATE (:RelationType {name: 'CHILD_OF', embedding: $e})", e=[0.0] * 63 + [1.0])

    response = client.get(f"/relations/connected/{harry}?type=child of", headers=auth_headers)

    assert response.json()["resolved_type"] == "PARENT_OF"
    assert [c["character"]["name"] for c in response.json()["connections"]] == ["Lily Potter"]


def test_blank_type_is_still_a_422(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.get("/relations/connected/any?type=%21%21", headers=auth_headers)

    assert response.status_code == 422


def test_unknown_type_makes_no_llm_call(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    seeded = seed(client, auth_headers, fake_llm)
    embeds, judges = len(fake_llm.embed_calls), len(fake_llm.relation_type_judge_calls)

    response = client.get(f"/relations/connected/{seeded['Harry Potter']['id']}?type=buddy of", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["resolved_type"] is None
    assert response.json()["connections"] == []
    assert len(fake_llm.embed_calls) == embeds
    assert len(fake_llm.relation_type_judge_calls) == judges


def test_type_without_of_suffix_is_resolved(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    seeded = seed(client, auth_headers, fake_llm)

    response = client.get(f"/relations/connected/{seeded['Harry Potter']['id']}?type=friend", headers=auth_headers)

    assert response.json()["resolved_type"] == "FRIEND_OF"
    assert len(response.json()["connections"]) == 2


def test_child_without_of_suffix_resolves_through_the_inverse_map(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    fake_llm.extraction = extraction(
        [character("Lily Potter", "Harry's mother"), character("Harry Potter", HARRY_DESCRIPTION)],
        [relation("Lily Potter", "Harry Potter", "PARENT_OF")],
    )
    body = client.post("/articles/upload", data={"text": "Lily and Harry."}, headers=auth_headers).json()
    harry = next(c["id"] for c in body["characters"] if c["name"] == "Harry Potter")

    response = client.get(f"/relations/connected/{harry}?type=child", headers=auth_headers)

    assert response.json()["resolved_type"] == "PARENT_OF"
    assert [c["character"]["name"] for c in response.json()["connections"]] == ["Lily Potter"]


def test_query_does_not_write(client: TestClient, auth_headers: dict[str, str], db: neo4j.Driver) -> None:
    insert_character(db, id="c1", name="Harry Potter")

    response = client.get("/relations/connected/c1?type=buddy of", headers=auth_headers)

    assert response.status_code == 200
    records, _, _ = db.execute_query("MATCH (t:RelationType) RETURN count(t) AS n")
    assert records[0]["n"] == 0


def test_query_works_without_llm_key(settings: Settings, db: neo4j.Driver) -> None:
    insert_character(db, id="c1", name="Harry Potter")

    with TestClient(create_app(settings, llm_client=UnconfiguredLLMClient())) as keyless:
        token = keyless.post("/auth/token", json={"app_token": APP_TOKEN}).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        query = keyless.get("/relations/connected/c1?type=buddy of", headers=headers)
        upload = keyless.post("/articles/upload", data={"text": "Some text."}, headers=headers)

    assert query.status_code == 200
    assert upload.status_code == 503

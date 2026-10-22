import asyncio
import hashlib
from collections.abc import AsyncIterator

import neo4j
import pytest
from fastapi.testclient import TestClient
from neo4j import AsyncGraphDatabase

from app.core.config import Settings
from app.graph.repository import ClaimLostError, GraphRepository, GraphWriter
from app.llm.vocabulary import CANONICAL_RELATION_TYPES
from app.main import create_app
from tests.factories import (
    character,
    extraction,
    harry_and_ron,
    insert_character,
    insert_processing_article,
    relation,
)
from tests.fakes import FakeLLMClient
from tests.integration.conftest import APP_TOKEN
from tests.integration.test_articles import MARKDOWN


def count(db: neo4j.Driver, query: str) -> int:
    records, _, _ = db.execute_query(query)
    return records[0]["n"]


def test_failed_write_leaves_nothing_behind(
    client: TestClient,
    auth_headers: dict[str, str],
    fake_llm: FakeLLMClient,
    db: neo4j.Driver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm.extraction = extraction(
        [character("Anna"), character("Boris")], [relation("Anna", "Boris", "IN_LOVE_WITH")]
    )

    async def fail(*_: object, **__: object) -> None:
        raise RuntimeError("neo4j down")

    monkeypatch.setattr(GraphWriter, "upsert_relations", fail)

    with pytest.raises(RuntimeError):
        client.post("/articles/upload", data={"text": MARKDOWN}, headers=auth_headers)

    assert count(db, "MATCH (a:Article) RETURN count(a) AS n") == 0
    assert count(db, "MATCH (c:Character) RETURN count(c) AS n") == 0
    records, _, _ = db.execute_query("MATCH (t:RelationType) RETURN t.name AS name")
    assert {r["name"] for r in records} == set(CANONICAL_RELATION_TYPES)


def upload_harry_and_ron(settings: Settings, db: neo4j.Driver) -> tuple[int, FakeLLMClient]:
    fake_llm = FakeLLMClient(settings.embedding_dimensions)
    fake_llm.extraction = harry_and_ron()
    with TestClient(create_app(settings, llm_client=fake_llm)) as client:
        token = client.post("/auth/token", json={"app_token": APP_TOKEN}).json()["access_token"]
        response = client.post(
            "/articles/upload", data={"text": MARKDOWN}, headers={"Authorization": f"Bearer {token}"}
        )
    return response.status_code, fake_llm


def test_concurrent_creation_of_the_same_character_is_merged(
    settings: Settings, db: neo4j.Driver, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, baseline = upload_harry_and_ron(settings, db)
    assert status == 201
    db.execute_query("MATCH (n) DETACH DELETE n")

    original = GraphRepository.write_transaction
    calls: list[int] = []

    async def competing_write(self: GraphRepository, work):
        if not calls:
            calls.append(1)
            insert_character(db, id="other", name="Harry Potter")
        return await original(self, work)

    monkeypatch.setattr(GraphRepository, "write_transaction", competing_write)

    status, fake_llm = upload_harry_and_ron(settings, db)

    assert status == 201
    records, _, _ = db.execute_query(
        "MATCH (c:Character {name_key: 'harry potter'}) "
        "RETURN c.id AS id, c.aliases AS aliases, size([(a:Article)-[:MENTIONS]->(c) | a]) AS mentions"
    )
    assert len(records) == 1
    assert records[0]["id"] == "other"
    assert "Harry" in records[0]["aliases"]
    assert records[0]["mentions"] == 1
    assert len(fake_llm.embed_calls) == len(baseline.embed_calls)


CONTENT_HASH = hashlib.sha256(MARKDOWN.encode()).hexdigest()


def upload(client: TestClient, headers: dict[str, str]):
    return client.post("/articles/upload", data={"text": MARKDOWN}, headers=headers)


def test_article_is_claimed_before_extraction(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    claimed: list[list[str]] = []

    def extract(_: str):
        records, _, _ = db.execute_query("MATCH (a:Article {status: 'processing'}) RETURN a.id AS id")
        claimed.append([r["id"] for r in records])
        return harry_and_ron()

    fake_llm.extraction = extract

    response = upload(client, auth_headers)

    assert response.status_code == 201
    assert claimed == [[response.json()["article_id"]]]


def test_in_progress_duplicate_returns_409_with_article_id(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    insert_processing_article(db, content_hash=CONTENT_HASH)

    response = upload(client, auth_headers)

    assert response.status_code == 409
    assert response.json()["article_id"] == "busy"
    assert fake_llm.extract_calls == []


def test_stale_processing_article_is_reprocessed(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.extraction = harry_and_ron()
    insert_processing_article(db, content_hash=CONTENT_HASH, age_minutes=40)

    response = upload(client, auth_headers)

    assert response.status_code == 201
    assert response.json()["article_id"] != "busy"
    assert count(db, "MATCH (a:Article {id: 'busy'}) RETURN count(a) AS n") == 0


def test_lost_claim_commits_nothing(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    def extract(_: str):
        # Another upload reclaims the article as stale while this one is still working.
        db.execute_query("MATCH (a:Article {status: 'processing'}) DETACH DELETE a")
        return harry_and_ron()

    fake_llm.extraction = extract

    response = upload(client, auth_headers)

    assert response.status_code in {409, 503}
    assert count(db, "MATCH (c:Character) RETURN count(c) AS n") == 0


def test_race_on_claim_returns_the_completed_duplicate(
    client: TestClient,
    auth_headers: dict[str, str],
    fake_llm: FakeLLMClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm.extraction = harry_and_ron()
    first = upload(client, auth_headers)
    original = GraphRepository.find_article_by_hash
    calls: list[int] = []

    async def blind_once(self: GraphRepository, content_hash: str):
        calls.append(1)
        return None if len(calls) == 1 else await original(self, content_hash)

    monkeypatch.setattr(GraphRepository, "find_article_by_hash", blind_once)

    second = upload(client, auth_headers)

    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["article_id"] == first.json()["article_id"]


@pytest.fixture
async def repository(settings: Settings, db: neo4j.Driver) -> AsyncIterator[GraphRepository]:
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value())
    )
    try:
        yield GraphRepository(driver, settings.neo4j_database, settings.embedding_dimensions)
    finally:
        await driver.close()


async def wait_until_blocked(task: asyncio.Task) -> None:
    # The statement waits for the write lock that the test's own transaction holds.
    await asyncio.sleep(1.5)
    assert not task.done()


async def test_stale_reclaim_waits_for_a_concurrent_commit_and_leaves_it_alone(
    repository: GraphRepository, db: neo4j.Driver
) -> None:
    insert_processing_article(db, content_hash=CONTENT_HASH, age_minutes=40)
    with db.session() as session:
        tx = session.begin_transaction()
        tx.run("MATCH (a:Article {id: 'busy'}) SET a.status = 'completed'").consume()
        reclaim = asyncio.create_task(repository.delete_unfinished_article("busy", older_than_seconds=1800))
        await wait_until_blocked(reclaim)
        tx.commit()
        await reclaim

    assert count(db, "MATCH (a:Article {id: 'busy', status: 'completed'}) RETURN count(a) AS n") == 1


async def test_commit_after_a_reclaim_deleted_the_claim_is_a_lost_claim(
    repository: GraphRepository, db: neo4j.Driver
) -> None:
    insert_processing_article(db, content_hash=CONTENT_HASH, age_minutes=40)
    with db.session() as session:
        tx = session.begin_transaction()
        tx.run("MATCH (a:Article {id: 'busy'}) DETACH DELETE a").consume()
        late_commit = asyncio.create_task(repository.write_transaction(lambda w: w.complete_article("busy", "t")))
        await wait_until_blocked(late_commit)
        tx.commit()
        with pytest.raises(ClaimLostError):
            await late_commit


async def test_fresh_claim_is_not_reclaimed(repository: GraphRepository, db: neo4j.Driver) -> None:
    insert_processing_article(db, content_hash=CONTENT_HASH, age_minutes=1)

    await repository.delete_unfinished_article("busy", older_than_seconds=1800)

    assert count(db, "MATCH (a:Article {id: 'busy'}) RETURN count(a) AS n") == 1

import os
from collections.abc import Iterator

import neo4j
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.fakes import FakeLLMClient

APP_TOKEN = "test-app-token"
NEO4J_IMAGE = "neo4j:5.26-community"
NEO4J_PASSWORD = "testpassword"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "integration" in item.path.parts:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def neo4j_connection() -> Iterator[tuple[str, str, str]]:
    # TEST_NEO4J_URI points tests at an existing instance. Every test wipes it, so never use a real one.
    external_uri = os.environ.get("TEST_NEO4J_URI")
    if external_uri:
        yield (
            external_uri,
            os.environ.get("TEST_NEO4J_USER", "neo4j"),
            os.environ.get("TEST_NEO4J_PASSWORD", NEO4J_PASSWORD),
        )
        return

    from testcontainers.community.neo4j import Neo4jContainer

    # Docker Desktop exposes a per-user socket path that cannot be bind-mounted into the Ryuk reaper.
    os.environ.setdefault("TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE", "/var/run/docker.sock")
    with Neo4jContainer(NEO4J_IMAGE, password=NEO4J_PASSWORD) as container:
        yield container.get_connection_url(), "neo4j", NEO4J_PASSWORD


@pytest.fixture(scope="session")
def neo4j_driver(neo4j_connection: tuple[str, str, str]) -> Iterator[neo4j.Driver]:
    uri, user, password = neo4j_connection
    with neo4j.GraphDatabase.driver(uri, auth=(user, password)) as driver:
        yield driver


@pytest.fixture
def db(neo4j_driver: neo4j.Driver) -> neo4j.Driver:
    """Sync driver used by tests to verify what the service persisted, independently of its own code."""
    neo4j_driver.execute_query("MATCH (n) DETACH DELETE n")
    return neo4j_driver


@pytest.fixture
def settings(neo4j_connection: tuple[str, str, str]) -> Settings:
    uri, user, password = neo4j_connection
    return Settings(
        _env_file=None,
        app_tokens=[APP_TOKEN],
        jwt_secret="test-secret-that-is-long-enough-for-hs256",
        neo4j_uri=uri,
        neo4j_user=user,
        neo4j_password=password,
        embedding_dimensions=64,
    )


@pytest.fixture
def fake_llm(settings: Settings) -> FakeLLMClient:
    return FakeLLMClient(settings.embedding_dimensions)


@pytest.fixture
def client(settings: Settings, fake_llm: FakeLLMClient, db: neo4j.Driver) -> Iterator[TestClient]:
    with TestClient(create_app(settings, llm_client=fake_llm)) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post("/auth/token", json={"app_token": APP_TOKEN})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}

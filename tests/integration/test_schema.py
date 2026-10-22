import neo4j
import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import ConstraintError

from app.core.config import Settings
from app.main import create_app
from tests.factories import insert_character
from tests.fakes import FakeLLMClient


def names(db: neo4j.Driver, query: str) -> set[str]:
    records, _, _ = db.execute_query(query)
    return {r["name"] for r in records}


def test_character_name_key_is_unique(client: TestClient, db: neo4j.Driver) -> None:
    assert "character_name_key_unique" in names(db, "SHOW CONSTRAINTS YIELD name")
    insert_character(db, id="a", name="Harry Potter")

    with pytest.raises(ConstraintError):
        insert_character(db, id="b", name="Harry  POTTER")


def test_schema_migrates_the_old_name_key_index(settings: Settings, fake_llm: FakeLLMClient, db: neo4j.Driver) -> None:
    db.execute_query("DROP CONSTRAINT character_name_key_unique IF EXISTS")
    db.execute_query("CREATE INDEX character_name_key IF NOT EXISTS FOR (c:Character) ON (c.name_key)")

    with TestClient(create_app(settings, llm_client=fake_llm)):
        pass

    assert "character_name_key_unique" in names(db, "SHOW CONSTRAINTS YIELD name")
    assert "character_name_key" not in names(db, "SHOW INDEXES YIELD name")

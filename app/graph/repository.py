from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import neo4j
from neo4j import AsyncDriver, RoutingControl
from neo4j.exceptions import ConstraintError

CHARACTER_VECTOR_INDEX = "character_embedding"
RELATION_TYPE_VECTOR_INDEX = "relation_type_embedding"

_CHARACTER_PROJECTION = "{.id, .name, .aliases, .description}"


class ArticleStatus(StrEnum):
    PROCESSING = "processing"
    COMPLETED = "completed"


class DuplicateArticleError(Exception):
    pass


Direction = Literal["any", "outgoing", "incoming"]


@dataclass(frozen=True)
class ArticleRecord:
    id: str
    title: str
    source_type: str
    status: ArticleStatus
    created_at_epoch: int


@dataclass(frozen=True)
class CharacterRecord:
    id: str
    name: str
    aliases: list[str]
    description: str

    @classmethod
    def from_map(cls, data: dict[str, Any]) -> "CharacterRecord":
        return cls(
            id=data["id"],
            name=data["name"],
            aliases=list(data.get("aliases") or []),
            description=data.get("description") or "",
        )


@dataclass(frozen=True)
class CharacterMatch:
    character: CharacterRecord
    score: float


@dataclass(frozen=True)
class RelationTypeMatch:
    name: str
    score: float


@dataclass(frozen=True)
class RelationRecord:
    source_id: str
    source_name: str
    target_id: str
    target_name: str
    type: str
    description: str


@dataclass(frozen=True)
class Connection:
    character: CharacterRecord
    direction: Literal["outgoing", "incoming"]
    description: str


class GraphRepository:
    def __init__(self, driver: AsyncDriver, database: str, embedding_dimensions: int) -> None:
        self._driver = driver
        self._database = database
        self._embedding_dimensions = embedding_dimensions

    async def _write(self, query: str, **params: Any) -> list[neo4j.Record]:
        result = await self._driver.execute_query(query, params, database_=self._database)
        return result.records

    async def _read(self, query: str, **params: Any) -> list[neo4j.Record]:
        result = await self._driver.execute_query(query, params, database_=self._database, routing_=RoutingControl.READ)
        return result.records

    async def ping(self) -> None:
        await self._read("RETURN 1")

    async def init_schema(self) -> None:
        # Schema commands do not accept parameters; the dimension is a validated int from settings.
        dimensions = int(self._embedding_dimensions)
        vector_options = (
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, `vector.similarity_function`: 'cosine'}}}}"
        )
        statements = [
            "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT article_content_hash IF NOT EXISTS FOR (a:Article) REQUIRE a.content_hash IS UNIQUE",
            "CREATE CONSTRAINT character_id IF NOT EXISTS FOR (c:Character) REQUIRE c.id IS UNIQUE",
            "CREATE INDEX character_name_key IF NOT EXISTS FOR (c:Character) ON (c.name_key)",
            "CREATE CONSTRAINT relation_type_name IF NOT EXISTS FOR (t:RelationType) REQUIRE t.name IS UNIQUE",
            "CREATE INDEX related_to_type IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.type)",
            f"CREATE VECTOR INDEX {CHARACTER_VECTOR_INDEX} IF NOT EXISTS "
            f"FOR (c:Character) ON c.embedding {vector_options}",
            f"CREATE VECTOR INDEX {RELATION_TYPE_VECTOR_INDEX} IF NOT EXISTS "
            f"FOR (t:RelationType) ON t.embedding {vector_options}",
        ]
        for statement in statements:
            await self._write(statement)
        await self._write("CALL db.awaitIndexes(300)")

    # Articles

    async def find_article_by_hash(self, content_hash: str) -> ArticleRecord | None:
        records = await self._read(
            "MATCH (a:Article {content_hash: $content_hash}) "
            "RETURN a.id AS id, a.title AS title, a.source_type AS source_type, a.status AS status, "
            "a.created_at.epochSeconds AS created_at_epoch",
            content_hash=content_hash,
        )
        if not records:
            return None
        record = records[0]
        return ArticleRecord(
            id=record["id"],
            title=record["title"],
            source_type=record["source_type"],
            status=ArticleStatus(record["status"]),
            created_at_epoch=record["created_at_epoch"],
        )

    async def create_article(
        self, *, article_id: str, title: str, source_type: str, content_hash: str, text: str
    ) -> None:
        try:
            await self._write(
                "CREATE (:Article {id: $id, title: $title, source_type: $source_type, "
                "content_hash: $content_hash, text: $text, status: $status, created_at: datetime()})",
                id=article_id,
                title=title,
                source_type=source_type,
                content_hash=content_hash,
                text=text,
                status=ArticleStatus.PROCESSING.value,
            )
        except ConstraintError as exc:
            raise DuplicateArticleError(content_hash) from exc

    async def mark_article_completed(self, article_id: str) -> None:
        await self._write(
            "MATCH (a:Article {id: $id}) SET a.status = $status, a.completed_at = datetime()",
            id=article_id,
            status=ArticleStatus.COMPLETED.value,
        )

    async def delete_article(self, article_id: str) -> None:
        await self._write(
            "MATCH (a:Article {id: $id}) "
            "OPTIONAL MATCH ()-[r:RELATED_TO]->() WHERE $id IN r.article_ids "
            "WITH a, collect(r) AS relations "
            "FOREACH (r IN [x IN relations WHERE size(x.article_ids) = 1] | DELETE r) "
            "FOREACH (r IN [x IN relations WHERE size(x.article_ids) > 1] | "
            "  SET r.article_ids = [i IN r.article_ids WHERE i <> $id]) "
            "DETACH DELETE a",
            id=article_id,
        )

    async def link_characters_to_article(self, article_id: str, character_ids: list[str]) -> None:
        await self._write(
            "MATCH (a:Article {id: $article_id}) "
            "UNWIND $character_ids AS character_id "
            "MATCH (c:Character {id: character_id}) "
            "MERGE (a)-[:MENTIONS]->(c)",
            article_id=article_id,
            character_ids=character_ids,
        )

    async def get_article_characters(self, article_id: str) -> list[CharacterRecord] | None:
        records = await self._read(
            "MATCH (a:Article {id: $id}) "
            "OPTIONAL MATCH (a)-[:MENTIONS]->(c:Character) "
            "WITH a, c ORDER BY c.name "
            # Grouping by a.id makes the query return no row (not an empty list) for a missing article.
            f"RETURN a.id AS id, collect(c {_CHARACTER_PROJECTION}) AS characters",
            id=article_id,
        )
        if not records:
            return None
        return [CharacterRecord.from_map(c) for c in records[0]["characters"]]

    async def get_article_relations(self, article_id: str) -> list[RelationRecord]:
        records = await self._read(
            "MATCH (s:Character)-[r:RELATED_TO]->(t:Character) WHERE $id IN r.article_ids "
            "RETURN s.id AS source_id, s.name AS source_name, t.id AS target_id, t.name AS target_name, "
            "r.type AS type, r.description AS description "
            "ORDER BY source_name, type, target_name",
            id=article_id,
        )
        return [RelationRecord(**record.data()) for record in records]

    # Characters

    async def find_character_by_name_key(self, key: str) -> CharacterRecord | None:
        records = await self._read(
            "MATCH (c:Character {name_key: $key}) WITH c ORDER BY c.created_at LIMIT 1 "
            f"RETURN c {_CHARACTER_PROJECTION} AS c",
            key=key,
        )
        return CharacterRecord.from_map(records[0]["c"]) if records else None

    async def find_characters_by_keys(self, keys: list[str]) -> list[CharacterRecord]:
        records = await self._read(
            "MATCH (c:Character) WHERE c.name_key IN $keys OR any(k IN c.alias_keys WHERE k IN $keys) "
            f"WITH c ORDER BY c.created_at RETURN c {_CHARACTER_PROJECTION} AS c",
            keys=keys,
        )
        return [CharacterRecord.from_map(r["c"]) for r in records]

    async def find_characters_by_tokens(self, tokens: list[str], limit: int) -> list[CharacterRecord]:
        """Characters whose name or any alias shares a whole word with `tokens`."""
        if not tokens:
            return []
        records = await self._read(
            "MATCH (c:Character) "
            "WHERE any(k IN [c.name_key] + coalesce(c.alias_keys, []) "
            "  WHERE any(t IN split(k, ' ') WHERE t IN $tokens)) "
            f"WITH c ORDER BY c.created_at LIMIT $limit RETURN c {_CHARACTER_PROJECTION} AS c",
            tokens=tokens,
            limit=limit,
        )
        return [CharacterRecord.from_map(r["c"]) for r in records]

    async def find_similar_characters(self, embedding: list[float], limit: int) -> list[CharacterMatch]:
        records = await self._read(
            "CALL db.index.vector.queryNodes($index, $limit, $embedding) YIELD node, score "
            f"RETURN node {_CHARACTER_PROJECTION} AS c, score",
            index=CHARACTER_VECTOR_INDEX,
            limit=limit,
            embedding=embedding,
        )
        return [CharacterMatch(CharacterRecord.from_map(r["c"]), r["score"]) for r in records]

    async def create_character(
        self, character: CharacterRecord, *, name_key: str, alias_keys: list[str], embedding: list[float]
    ) -> None:
        await self._write(
            "CREATE (:Character {id: $id, name: $name, name_key: $name_key, aliases: $aliases, "
            "alias_keys: $alias_keys, description: $description, embedding: $embedding, created_at: datetime()})",
            id=character.id,
            name=character.name,
            name_key=name_key,
            aliases=character.aliases,
            alias_keys=alias_keys,
            description=character.description,
            embedding=embedding,
        )

    async def update_character(self, character: CharacterRecord, *, alias_keys: list[str]) -> None:
        await self._write(
            "MATCH (c:Character {id: $id}) "
            "SET c.aliases = $aliases, c.alias_keys = $alias_keys, c.description = $description",
            id=character.id,
            aliases=character.aliases,
            alias_keys=alias_keys,
            description=character.description,
        )

    async def list_characters(self, limit: int, offset: int) -> list[CharacterRecord]:
        records = await self._read(
            f"MATCH (c:Character) RETURN c {_CHARACTER_PROJECTION} AS c ORDER BY c.name, c.id "
            "SKIP $offset LIMIT $limit",
            offset=offset,
            limit=limit,
        )
        return [CharacterRecord.from_map(r["c"]) for r in records]

    # Relations

    async def upsert_relations(self, article_id: str, relations: list[dict[str, str]]) -> None:
        await self._write(
            "UNWIND $relations AS rel "
            "MATCH (s:Character {id: rel.source_id}), (t:Character {id: rel.target_id}) "
            "MERGE (s)-[r:RELATED_TO {type: rel.type}]->(t) "
            "ON CREATE SET r.description = rel.description, r.article_ids = [$article_id], "
            "  r.created_at = datetime() "
            "ON MATCH SET r.article_ids = CASE WHEN $article_id IN r.article_ids THEN r.article_ids "
            "  ELSE r.article_ids + $article_id END",
            article_id=article_id,
            relations=relations,
        )

    async def list_relation_types(self) -> list[tuple[str, int]]:
        records = await self._read(
            "MATCH ()-[r:RELATED_TO]->() RETURN r.type AS type, count(r) AS count ORDER BY count DESC, type"
        )
        return [(r["type"], r["count"]) for r in records]

    async def get_connections(
        self, character_id: str, relation_type: str, direction: Direction
    ) -> list[Connection] | None:
        records = await self._read(
            "MATCH (c:Character {id: $id}) "
            "OPTIONAL MATCH (c)-[r:RELATED_TO {type: $type}]-(other:Character) "
            "WHERE $direction = 'any' "
            "   OR ($direction = 'outgoing' AND startNode(r) = c) "
            "   OR ($direction = 'incoming' AND endNode(r) = c) "
            "WITH c, r, other ORDER BY other.name "
            "RETURN c.id AS id, collect(CASE WHEN other IS NULL THEN NULL ELSE {"
            f"  character: other {_CHARACTER_PROJECTION}, "
            "  direction: CASE WHEN startNode(r) = c THEN 'outgoing' ELSE 'incoming' END, "
            "  description: r.description} END) AS connections",
            id=character_id,
            type=relation_type,
            direction=direction,
        )
        if not records:
            return None
        return [
            Connection(
                character=CharacterRecord.from_map(item["character"]),
                direction=item["direction"],
                description=item["description"] or "",
            )
            for item in records[0]["connections"]
        ]

    # Relation types

    async def existing_relation_types(self, names: list[str]) -> set[str]:
        records = await self._read("MATCH (t:RelationType) WHERE t.name IN $names RETURN t.name AS name", names=names)
        return {r["name"] for r in records}

    async def find_similar_relation_types(self, embedding: list[float], limit: int) -> list[RelationTypeMatch]:
        records = await self._read(
            "CALL db.index.vector.queryNodes($index, $limit, $embedding) YIELD node, score "
            "RETURN node.name AS name, score",
            index=RELATION_TYPE_VECTOR_INDEX,
            limit=limit,
            embedding=embedding,
        )
        return [RelationTypeMatch(r["name"], r["score"]) for r in records]

    async def create_relation_types(self, items: list[tuple[str, list[float]]]) -> None:
        if not items:
            return
        await self._write(
            "UNWIND $items AS item "
            "MERGE (t:RelationType {name: item.name}) "
            "ON CREATE SET t.embedding = item.embedding, t.created_at = datetime()",
            items=[{"name": name, "embedding": embedding} for name, embedding in items],
        )

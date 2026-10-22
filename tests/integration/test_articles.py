import logging

import neo4j
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.llm.base import LLMError, LLMNotConfiguredError, LLMUnavailableError
from app.llm.schemas import ImageAnalysis
from app.llm.vocabulary import CANONICAL_RELATION_TYPES
from app.main import create_app
from tests.factories import (
    HARRY_DESCRIPTION,
    HERMIONE_DESCRIPTION,
    RON_DESCRIPTION,
    build_pdf,
    character,
    extraction,
    harry_and_ron,
    insert_processing_article,
    relation,
)
from tests.fakes import FakeLLMClient

MARKDOWN = "# Chapter One\n\nHarry Potter met Ron Weasley on the Hogwarts Express."


def upload_markdown(client: TestClient, headers: dict[str, str], text: str = MARKDOWN) -> dict:
    response = client.post("/articles/upload", data={"text": text}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


def test_markdown_upload_persists_article_characters_and_relations(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.extraction = harry_and_ron()

    body = upload_markdown(client, auth_headers)

    assert body["title"] == "Chapter One"
    assert body["source_type"] == "markdown"
    assert body["duplicate"] is False
    assert {c["name"] for c in body["characters"]} == {"Harry Potter", "Ron Weasley"}
    assert [(r["source_name"], r["type"], r["target_name"]) for r in body["relations"]] == [
        ("Harry Potter", "FRIEND_OF", "Ron Weasley")
    ]

    records, _, _ = db.execute_query(
        "MATCH (a:Article {id: $id})-[:MENTIONS]->(c:Character) RETURN c.name AS name, c.embedding AS e",
        id=body["article_id"],
    )
    assert {r["name"] for r in records} == {"Harry Potter", "Ron Weasley"}
    assert all(len(r["e"]) == fake_llm.embedding_dimensions for r in records)

    records, _, _ = db.execute_query(
        "MATCH (s:Character)-[r:RELATED_TO]->(t:Character) "
        "RETURN s.name AS s, r.type AS type, t.name AS t, r.article_ids AS articles"
    )
    assert [(r["s"], r["type"], r["t"], r["articles"]) for r in records] == [
        ("Harry Potter", "FRIEND_OF", "Ron Weasley", [body["article_id"]])
    ]

    records, _, _ = db.execute_query("MATCH (t:RelationType) RETURN t.name AS name")
    assert {r["name"] for r in records} == set(CANONICAL_RELATION_TYPES)


def test_markdown_file_upload(client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient) -> None:
    fake_llm.extraction = harry_and_ron()

    response = client.post(
        "/articles/upload",
        files={"file": ("chapter.md", MARKDOWN.encode(), "text/markdown")},
        headers=auth_headers,
    )

    assert response.status_code == 201, response.text
    assert response.json()["source_type"] == "markdown"
    assert fake_llm.extract_calls[0][0].startswith("# Chapter One")


def test_pdf_upload_sends_images_to_vision_model_and_uses_descriptions(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.image_analysis = ImageAnalysis(
        is_relevant=True, description="Family tree showing Hermione Granger as a friend of Harry Potter."
    )
    fake_llm.extraction = extraction(
        [character("Harry Potter", HARRY_DESCRIPTION), character("Hermione Granger", HERMIONE_DESCRIPTION)],
        [relation("Hermione Granger", "Harry Potter", "FRIEND_OF")],
    )
    pdf = build_pdf([{"text": "Harry Potter went to Hogwarts.", "image_size": (300, 300)}])

    response = client.post(
        "/articles/upload", files={"file": ("book.pdf", pdf, "application/pdf")}, headers=auth_headers
    )

    assert response.status_code == 201, response.text
    assert response.json()["source_type"] == "pdf"
    assert len(fake_llm.image_calls) == 1
    extracted_text = fake_llm.extract_calls[0][0]
    assert "Harry Potter went to Hogwarts." in extracted_text
    assert "Family tree showing Hermione Granger" in extracted_text
    records, _, _ = db.execute_query("MATCH (c:Character) RETURN c.name AS name ORDER BY name")
    assert [r["name"] for r in records] == ["Harry Potter", "Hermione Granger"]


def test_irrelevant_images_are_not_added_to_text(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    fake_llm.image_analysis = ImageAnalysis(is_relevant=False, description="A decorative border.")
    pdf = build_pdf([{"text": "Some text.", "image_size": (300, 300)}])

    client.post("/articles/upload", files={"file": ("a.pdf", pdf, "application/pdf")}, headers=auth_headers)

    assert "decorative border" not in fake_llm.extract_calls[0][0]


def test_same_content_uploaded_twice_is_not_processed_again(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    fake_llm.extraction = harry_and_ron()
    first = upload_markdown(client, auth_headers)

    response = client.post("/articles/upload", data={"text": MARKDOWN}, headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["duplicate"] is True
    assert body["article_id"] == first["article_id"]
    assert {c["name"] for c in body["characters"]} == {"Harry Potter", "Ron Weasley"}
    assert len(body["relations"]) == 1
    assert len(fake_llm.extract_calls) == 1


def test_characters_are_deduplicated_across_articles_by_alias(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.extraction = harry_and_ron()
    upload_markdown(client, auth_headers)
    fake_llm.extraction = extraction(
        [character("Harry", "The boy with the scar"), character("Ron", "Loyal friend")],
        [relation("Ron", "Harry", "FRIEND_OF")],
    )
    fake_llm.same_characters = [{"Harry", "Harry Potter"}, {"Ron", "Ron Weasley"}]

    upload_markdown(client, auth_headers, text="Ron and Harry played chess.")

    records, _, _ = db.execute_query(
        "MATCH (c:Character) OPTIONAL MATCH (a:Article)-[:MENTIONS]->(c) "
        "RETURN c.name AS name, count(a) AS articles ORDER BY name"
    )
    assert [(r["name"], r["articles"]) for r in records] == [("Harry Potter", 2), ("Ron Weasley", 2)]


def test_characters_are_deduplicated_across_articles_via_judge(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.extraction = extraction([character("Harry Potter", HARRY_DESCRIPTION)])
    upload_markdown(client, auth_headers)
    fake_llm.extraction = extraction(
        [character("The Boy Who Lived", HARRY_DESCRIPTION), character("Ronald Weasley", RON_DESCRIPTION)]
    )
    fake_llm.same_characters = [{"Harry Potter", "The Boy Who Lived"}]

    upload_markdown(client, auth_headers, text="The Boy Who Lived and Ronald Weasley.")

    records, _, _ = db.execute_query("MATCH (c:Character) RETURN c.name AS name, c.aliases AS aliases")
    by_name = {r["name"]: r["aliases"] for r in records}
    assert set(by_name) == {"Harry Potter", "Ronald Weasley"}
    assert "The Boy Who Lived" in by_name["Harry Potter"]


def test_extraction_receives_vocabulary_and_used_types(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    db.execute_query("CREATE (:RelationType {name: 'ORPHAN_TYPE', embedding: $e})", e=[0.0] * 63 + [1.0])
    fake_llm.extraction = extraction(
        [character("Harry Potter", HARRY_DESCRIPTION), character("Ron Weasley", RON_DESCRIPTION)],
        [relation("Harry Potter", "Ron Weasley", "IN_LOVE_WITH")],
    )
    upload_markdown(client, auth_headers)

    upload_markdown(client, auth_headers, text="Harry and Ron again.")

    assert fake_llm.extract_calls[1][1] == [*CANONICAL_RELATION_TYPES, "IN_LOVE_WITH"]
    assert "ORPHAN_TYPE" not in fake_llm.extract_calls[1][1]


def test_relations_with_unknown_characters_or_self_loops_are_dropped(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.extraction = extraction(
        [character("Harry Potter", HARRY_DESCRIPTION, aliases=("Harry",))],
        [relation("Harry Potter", "Voldemort", "ENEMY_OF"), relation("Harry Potter", "Harry", "SELF")],
    )

    body = upload_markdown(client, auth_headers)

    assert body["relations"] == []
    records, _, _ = db.execute_query("MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS n")
    assert records[0]["n"] == 0


def test_upload_without_content_is_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post("/articles/upload", headers=auth_headers)

    assert response.status_code == 422


def test_upload_with_both_text_and_file_is_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/articles/upload",
        data={"text": MARKDOWN},
        files={"file": ("a.md", b"# x", "text/markdown")},
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_unsupported_file_type_is_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/articles/upload", files={"file": ("a.docx", b"PK\x03\x04", "application/msword")}, headers=auth_headers
    )

    assert response.status_code == 415


def test_corrupted_pdf_is_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/articles/upload", files={"file": ("a.pdf", b"%PDF-1.7 garbage", "application/pdf")}, headers=auth_headers
    )

    assert response.status_code == 422


def test_failed_extraction_leaves_no_article(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    def fail(_: str):
        raise LLMError("model refused")

    fake_llm.extraction = fail

    response = client.post("/articles/upload", data={"text": MARKDOWN}, headers=auth_headers)

    assert response.status_code == 502
    records, _, _ = db.execute_query("MATCH (a:Article) RETURN count(a) AS n")
    assert records[0]["n"] == 0


def test_llm_error_text_is_not_returned_to_the_client(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    def fail(_: str):
        raise LLMError("org-secret-123")

    fake_llm.extraction = fail

    response = client.post("/articles/upload", data={"text": MARKDOWN}, headers=auth_headers)

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM backend error"}
    assert "org-secret-123" not in response.text


def upload_pdf(client: TestClient, headers: dict[str, str], pdf: bytes, title: str | None = None):
    return client.post(
        "/articles/upload",
        files={"file": ("book.pdf", pdf, "application/pdf")},
        data={"title": title} if title else None,
        headers=headers,
    )


def article_count(db: neo4j.Driver) -> int:
    records, _, _ = db.execute_query("MATCH (a:Article) RETURN count(a) AS n")
    return records[0]["n"]


def test_one_failed_visual_is_skipped(
    client: TestClient,
    auth_headers: dict[str, str],
    fake_llm: FakeLLMClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_llm.image_results = [
        LLMError("refused"),
        ImageAnalysis(is_relevant=True, description="Chart: Anna mother of Boris"),
    ]
    fake_llm.extraction = harry_and_ron()
    pdf = build_pdf(
        [
            {"text": "Page one text.", "image_size": (300, 300)},
            {"text": "Page two text.", "image_size": (320, 310)},
        ]
    )

    with caplog.at_level(logging.WARNING, logger="app.pipeline.ingestion"):
        response = upload_pdf(client, auth_headers, pdf, title="book")

    assert response.status_code == 201, response.text
    assert "Chart: Anna mother of Boris" in fake_llm.extract_calls[0][0]
    assert "book" in caplog.text and "page 1" in caplog.text and "skipped" in caplog.text
    assert "1 of 2 visuals skipped" in caplog.text


def test_failed_only_visual_with_page_text_is_skipped(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    fake_llm.image_results = [LLMError("x")]
    fake_llm.extraction = harry_and_ron()
    pdf = build_pdf([{"text": "Only the page text.", "image_size": (300, 300)}])

    response = upload_pdf(client, auth_headers, pdf)

    assert response.status_code == 201, response.text
    assert fake_llm.extract_calls[0][0] == "Only the page text."


def test_all_visuals_failing_without_text_returns_502(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.image_results = [LLMError("x")]
    pdf = build_pdf([{"image_size": (600, 800)}])

    response = upload_pdf(client, auth_headers, pdf)

    assert response.status_code == 502
    assert article_count(db) == 0


def test_unconfigured_vision_is_still_503(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    fake_llm.image_results = [LLMNotConfiguredError("x")]
    pdf = build_pdf([{"text": "Some text.", "image_size": (300, 300)}])

    assert upload_pdf(client, auth_headers, pdf).status_code == 503


def test_too_many_chunks_is_rejected_before_any_llm_call(
    settings: Settings, fake_llm: FakeLLMClient, db: neo4j.Driver, auth_headers: dict[str, str]
) -> None:
    limited = settings.model_copy(update={"max_chunks": 1, "chunk_size_chars": 100, "chunk_overlap_chars": 10})
    text = "\n\n".join(f"Paragraph {i} " + "word " * 14 for i in range(3))

    with TestClient(create_app(limited, llm_client=fake_llm)) as limited_client:
        response = limited_client.post("/articles/upload", data={"text": text}, headers=auth_headers)

    assert response.status_code == 422
    assert "MAX_CHUNKS" in response.json()["detail"]
    assert fake_llm.extract_calls == []
    assert article_count(db) == 0


def friend_edges(db: neo4j.Driver) -> list[tuple[str, str, list[str]]]:
    records, _, _ = db.execute_query(
        "MATCH (s:Character)-[r:RELATED_TO {type: 'FRIEND_OF'}]->(t:Character) "
        "RETURN s.name AS s, t.name AS t, r.article_ids AS articles"
    )
    return [(r["s"], r["t"], r["articles"]) for r in records]


def test_symmetric_relation_stated_in_both_directions_is_stored_once(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.extraction = extraction(
        [character("Harry Potter", HARRY_DESCRIPTION), character("Ron Weasley", RON_DESCRIPTION)],
        [relation("Harry Potter", "Ron Weasley", "FRIEND_OF"), relation("Ron Weasley", "Harry Potter", "FRIEND_OF")],
    )

    body = upload_markdown(client, auth_headers)

    assert [(r["source_name"], r["target_name"]) for r in body["relations"]] == [("Harry Potter", "Ron Weasley")]
    assert friend_edges(db) == [("Harry Potter", "Ron Weasley", [body["article_id"]])]


def test_symmetric_relation_is_not_duplicated_by_a_later_article_in_the_reverse_direction(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.extraction = harry_and_ron()
    first = upload_markdown(client, auth_headers)
    fake_llm.extraction = extraction(
        [character("Harry Potter", HARRY_DESCRIPTION), character("Ron Weasley", RON_DESCRIPTION)],
        [relation("Ron Weasley", "Harry Potter", "FRIEND_OF")],
    )
    fake_llm.same_characters = [{"Harry Potter"}, {"Ron Weasley"}]

    second = upload_markdown(client, auth_headers, text="Ron and Harry played chess.")

    assert friend_edges(db) == [("Harry Potter", "Ron Weasley", [first["article_id"], second["article_id"]])]


def test_synonym_mapped_to_a_symmetric_type_dedupes_against_the_reverse_edge(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.embedding_overrides = {"best friends with": "friend of"}
    fake_llm.same_relation_types = {("BEST_FRIENDS_WITH", "FRIEND_OF")}
    fake_llm.extraction = extraction(
        [character("Harry Potter", HARRY_DESCRIPTION), character("Ron Weasley", RON_DESCRIPTION)],
        [
            relation("Harry Potter", "Ron Weasley", "FRIEND_OF"),
            relation("Ron Weasley", "Harry Potter", "BEST_FRIENDS_WITH"),
        ],
    )

    body = upload_markdown(client, auth_headers)

    assert friend_edges(db) == [("Harry Potter", "Ron Weasley", [body["article_id"]])]


def test_get_article_returns_status_and_result(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient
) -> None:
    fake_llm.extraction = harry_and_ron()
    uploaded = upload_markdown(client, auth_headers)

    response = client.get(f"/articles/{uploaded['article_id']}", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["characters"] == uploaded["characters"]
    assert body["relations"] == uploaded["relations"]


def test_get_processing_article(client: TestClient, auth_headers: dict[str, str], db: neo4j.Driver) -> None:
    insert_processing_article(db)

    body = client.get("/articles/busy", headers=auth_headers).json()

    assert body["status"] == "processing"
    assert body["characters"] == []
    assert body["relations"] == []


def test_get_stale_article(client: TestClient, auth_headers: dict[str, str], db: neo4j.Driver) -> None:
    insert_processing_article(db, age_minutes=40)

    assert client.get("/articles/busy", headers=auth_headers).json()["status"] == "stale"


def test_get_unknown_article_returns_404(client: TestClient, auth_headers: dict[str, str]) -> None:
    assert client.get("/articles/missing", headers=auth_headers).status_code == 404


def test_oversized_upload_returns_413(
    settings: Settings, fake_llm: FakeLLMClient, db: neo4j.Driver, auth_headers: dict[str, str]
) -> None:
    small = settings.model_copy(update={"max_upload_bytes": 10})

    with TestClient(create_app(small, llm_client=fake_llm)) as small_client:
        response = small_client.post("/articles/upload", data={"text": "x" * 11}, headers=auth_headers)

    assert response.status_code == 413


def test_embedding_count_mismatch_on_upload_returns_502(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.extraction = harry_and_ron()

    async def broken_embed(texts: list[str]) -> list[list[float]]:
        return []

    fake_llm.embed = broken_embed  # type: ignore[method-assign]

    response = client.post("/articles/upload", data={"text": MARKDOWN}, headers=auth_headers)

    assert response.status_code == 502
    assert article_count(db) == 0


def test_vision_outage_fails_the_upload_and_releases_the_claim(
    client: TestClient, auth_headers: dict[str, str], fake_llm: FakeLLMClient, db: neo4j.Driver
) -> None:
    fake_llm.image_results = [LLMUnavailableError("rate limited")]
    fake_llm.extraction = harry_and_ron()
    pdf = build_pdf([{"text": "Page text.", "image_size": (300, 300)}])

    response = upload_pdf(client, auth_headers, pdf)

    assert response.status_code == 502
    assert fake_llm.extract_calls == []
    assert article_count(db) == 0

    fake_llm.image_results = [ImageAnalysis(is_relevant=True, description="Chart: Anna mother of Boris")]
    assert upload_pdf(client, auth_headers, pdf).status_code == 201

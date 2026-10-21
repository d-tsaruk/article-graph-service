# Article Graph Service

A FastAPI microservice that reads articles (Markdown or PDF, including their images and graphics), extracts
characters and the relations between them with an LLM, and stores everything in a Neo4j graph. Neo4j's
built-in vector index is used for entity and relation-type deduplication and for semantic lookup of relation types.

- **API:** FastAPI, JWT auth (App Token → access + refresh)
- **LLM:** OpenAI or any OpenAI-compatible API, Structured Outputs validated by Pydantic
- **PDF:** PyMuPDF for text, embedded images, scanned pages and vector graphics; a vision model describes the visuals
- **Storage:** Neo4j 5 (graph + native HNSW vector indexes)

## Quick start

```bash
cp .env.example .env          # put your OPENAI_API_KEY there
docker compose up -d          # or: docker-compose up -d
open http://localhost:8000/docs
```

Without `.env` the stack still starts with development defaults: the read endpoints work and uploads return `503`
until `OPENAI_API_KEY` is set. Neo4j Browser is available at http://localhost:7474 (`neo4j` / `local-dev-password`).

## Architecture

```
                   ┌──────────────────────────── FastAPI (app/) ────────────────────────────┐
 client ──JWT──▶   │  api/routes ──▶ pipeline/ingestion.ArticleIngestionService             │
                   │                   │                                                    │
                   │                   ├─ 1. documents.parse_pdf / parse_markdown (PyMuPDF) │
                   │                   ├─ 2. LLM vision: describe images & rendered pages   │
                   │                   ├─ 3. chunking.chunk_text (paragraph packing+overlap)│
                   │                   ├─ 4. LLM extraction per chunk (Structured Outputs)  │
                   │                   ├─ 5. merging.merge_extractions (union-find)         │
                   │                   ├─ 6. resolution: dedup vs. graph (vector index)     │
                   │                   └─ 7. graph/repository: persist                      │
                   └───────────────┬───────────────────────────────┬────────────────────────┘
                                   │ Bolt                          │ HTTPS
                          ┌────────▼────────┐             ┌────────▼────────┐
                          │ Neo4j 5         │             │ OpenAI-compat.  │
                          │ graph + vectors │             │ chat / vision / │
                          └─────────────────┘             │ embeddings      │
                                                          └─────────────────┘
```

### Ingestion pipeline

1. **Content hash.** SHA-256 of the uploaded bytes. Re-uploading identical content returns the stored result
   (`200`, `"duplicate": true`) without calling the LLM.
2. **Parsing.**
   - Markdown is passed to the model as is. Markdown structure helps it, and the title comes from the first heading.
   - PDF text is read per page in reading order. Visual content is collected per page:
     - **Embedded raster images** of at least 100 px. Each unique image (by xref) is analysed once, even if it
       repeats on several pages. Formats the vision API does not accept (JPX, JBIG2, CMYK) are re-encoded to PNG.
     - **Scanned pages** (large image and almost no text layer) are rendered to PNG as a whole, so the vision
       model OCRs them.
     - **Vector graphics** (charts and diagrams drawn as paths, which are not images) are detected by the number
       of drawing operations. Such a page is rendered and analysed as a whole.
     - The number of vision calls per document is capped (`PDF_MAX_VISUAL_ITEMS`).
3. **Vision.** Every visual goes to the vision model, with the page text as context. The model returns
   `{is_relevant, description}` (Structured Output). Decorative images are dropped. Relevant descriptions are
   appended to the page text as `[Visual content: …]`, so relations shown only in a diagram (for example a family
   tree) are extracted like any other text.
4. **Chunking.** Paragraph-aware packing (`CHUNK_SIZE_CHARS`, default 12 000) with overlapping trailing
   paragraphs, so that relations described across a boundary are seen together.
5. **Extraction.** Chunks are processed concurrently, bounded by `LLM_CONCURRENCY`. Each call uses OpenAI
   Structured Outputs (`chat.completions.parse` with a Pydantic `response_format`, strict JSON schema), so the
   response is validated before the pipeline sees it. Refusals, truncation and API errors become `502`. Relation
   types already present in the graph are passed to the prompt as a preferred vocabulary.
6. **In-article merge.** Characters from all chunks are unified with union-find over normalized names and aliases.
   Relations are rewritten to canonical names, typed (`UPPER_SNAKE_CASE`) and deduplicated. Relations that point to
   unknown characters, and self-loops, are dropped.
7. **Resolution against the graph** (see [Deduplication](#deduplication-and-the-vector-index)).
8. **Persistence.** The article is created as `processing`. Characters, mentions and relations are written, then
   the article is marked `completed`. Steps 1–6 have no side effects, and a failure in step 8 deletes the article,
   so a failed upload does not leave a half-written article behind.

### Deduplication and the vector index

Both indexes are Neo4j native vector indexes (HNSW, cosine). Neo4j reports cosine similarity as
`score = (1 + cos) / 2` ∈ [0, 1]; all thresholds below are on that scale.

| Index                     | Node / property          | Embedded text                         | Used for                                 |
|---------------------------|--------------------------|---------------------------------------|------------------------------------------|
| `character_embedding`     | `Character.embedding`    | `name (also known as …): description` | merging a new character into a known one |
| `relation_type_embedding` | `RelationType.embedding` | `friend of` (from `FRIEND_OF`)        | synonym types on write and on query      |

**Characters.** Each extracted character is matched in order:

1. Exact match of any normalized name or alias (case, Unicode width and punctuation are ignored).
2. Otherwise, the top 5 vector neighbours. A candidate is accepted when
   - `score ≥ CHARACTER_AUTO_MERGE_SCORE` (0.97), or
   - `score ≥ CHARACTER_MATCH_SCORE` (0.93) **and** the names share an identifying token (honorifics such as
     `Mr`, `Professor`, `Lord` are ignored). So `Mr. Potter` merges into `Harry Potter`, while two different
     `Dursley`s with different descriptions do not.
3. Otherwise a new character is created. A merged name becomes an alias of the stored character.

Characters are resolved one by one on purpose: a character created a moment ago must already be visible
to the vector index when the next one is resolved.

**Relation types.** On write, a new type such as `FRIENDS_WITH` reuses the closest stored type such as `FRIEND_OF`
when `score ≥ RELATION_TYPE_MATCH_SCORE` (0.95), so the vocabulary does not fragment into synonyms. On read,
`GET /relations/connected/{id}?type=buddy of` normalizes the input to `BUDDY_OF`, then tries an exact match, then
the nearest stored type. The response shows what was used in `resolved_type`.

### Graph schema

```
(:Article {id, title, source_type, content_hash, text, status, created_at, completed_at})
    -[:MENTIONS]->
(:Character {id, name, name_key, aliases[], alias_keys[], description, embedding[], created_at})
    -[:RELATED_TO {type, description, article_ids[], created_at}]->
(:Character)

(:RelationType {name, embedding[], created_at})      // vocabulary of relation types
```

| Element              | Constraint / index                                            |
|----------------------|---------------------------------------------------------------|
| `Article.id`         | unique                                                        |
| `Article.content_hash` | unique (idempotent uploads, protects against parallel duplicates) |
| `Character.id`       | unique                                                        |
| `Character.name_key` | range index (exact name lookup)                               |
| `Character.embedding` | vector index `character_embedding`                           |
| `RelationType.name`  | unique                                                        |
| `RelationType.embedding` | vector index `relation_type_embedding`                    |
| `RELATED_TO.type`    | relationship property index                                   |

Design notes:

- **The relation type is a property, not a Neo4j relationship type.** The vocabulary comes from the LLM and is
  open-ended. Cypher cannot parameterize relationship types, so a dynamic type would mean string interpolation or
  APOC. A property keeps every query parameterized and is indexed.
- **Relations are directed** (`PARENT_OF` goes from parent to child). `connected` takes
  `direction=any|outgoing|incoming`. One edge per `(source, type, target)` is kept, and `article_ids` lists the
  articles where it was found.
- IDs are UUIDs.

## API

| Method | Path                                         | Auth    | Description                                         |
|--------|----------------------------------------------|---------|-----------------------------------------------------|
| POST   | `/auth/token`                                | —       | App Token → `{access_token, refresh_token}`         |
| POST   | `/auth/refresh`                              | —       | refresh token → new pair                            |
| POST   | `/articles/upload`                           | Bearer  | multipart: `text` (Markdown) **or** `file` (.md/.pdf), optional `title` |
| GET    | `/characters?limit=&offset=`                 | Bearer  | all characters, sorted by name                      |
| GET    | `/characters/by_article/{article_id}`        | Bearer  | characters mentioned in an article (`404` if unknown) |
| GET    | `/relations/types`                           | Bearer  | unique relation types with counts                   |
| GET    | `/relations/connected/{character_id}?type=&direction=` | Bearer | characters connected by a relation type |
| GET    | `/health`                                    | —       | liveness + Neo4j connectivity                       |

Interactive docs: http://localhost:8000/docs.

Upload error codes: `413` file too large (`MAX_UPLOAD_BYTES`, 20 MB by default), `415` unsupported file type,
`422` unreadable or empty document, `409` identical content is being processed right now, `502` LLM backend error,
`503` LLM not configured.

### Authentication

App Tokens are configured in `APP_TOKENS` (comma-separated). They represent client applications, for example the
main Django project. A client exchanges its App Token for a short-lived **access** token (15 min) and a
**refresh** token (24 h). Both are HS256 JWTs with a `type` claim, so neither can be used in place of the other.
Tokens carry a `jti` so that a revocation list can be added later; the prototype is stateless.

### curl examples

```bash
BASE=http://localhost:8000

# 1. Exchange the App Token for a token pair
TOKENS=$(curl -s -X POST $BASE/auth/token \
  -H 'Content-Type: application/json' \
  -d '{"app_token": "local-dev-app-token"}')
echo "$TOKENS"
# {"access_token":"eyJ...","refresh_token":"eyJ...","token_type":"bearer","expires_in":900,"refresh_expires_in":86400}

ACCESS=$(echo "$TOKENS" | python3 -c 'import sys, json; print(json.load(sys.stdin)["access_token"])')
REFRESH=$(echo "$TOKENS" | python3 -c 'import sys, json; print(json.load(sys.stdin)["refresh_token"])')
AUTH="Authorization: Bearer $ACCESS"

# 2. Refresh when the access token expires
curl -s -X POST $BASE/auth/refresh -H 'Content-Type: application/json' -d "{\"refresh_token\": \"$REFRESH\"}"

# 3a. Upload Markdown text
curl -s -X POST $BASE/articles/upload -H "$AUTH" \
  -F 'text=# The Boy Who Lived

Harry Potter lived with his aunt Petunia Dursley and uncle Vernon Dursley.
On the Hogwarts Express Harry met Ron Weasley, who became his best friend.
Hermione Granger soon became friends with both of them.'

# 3b. Upload a Markdown or PDF file (title is optional)
curl -s -X POST $BASE/articles/upload -H "$AUTH" -F 'file=@article.md'
curl -s -X POST $BASE/articles/upload -H "$AUTH" -F 'file=@book.pdf' -F 'title=Chapter 1'
# {"article_id":"6f1c…","title":"The Boy Who Lived","source_type":"markdown","duplicate":false,
#  "characters":[{"id":"…","name":"Harry Potter","aliases":["Harry"],"description":"…"}, …],
#  "relations":[{"source_id":"…","source_name":"Harry Potter","target_id":"…","target_name":"Ron Weasley",
#                "type":"FRIEND_OF","description":"…"}, …]}

# 4. All characters / characters of an article
curl -s $BASE/characters -H "$AUTH"
curl -s $BASE/characters/by_article/<article_id> -H "$AUTH"

# 5. Relation types
curl -s $BASE/relations/types -H "$AUTH"
# [{"type":"FRIEND_OF","count":3},{"type":"NEPHEW_OF","count":2}]

# 6. Who is a friend of the character? (`type` accepts FRIEND_OF, "friend of", or a synonym)
curl -s -G $BASE/relations/connected/<character_id> -H "$AUTH" --data-urlencode 'type=friend of'
curl -s -G $BASE/relations/connected/<character_id> -H "$AUTH" -d type=PARENT_OF -d direction=outgoing
# {"character_id":"…","requested_type":"friend of","resolved_type":"FRIEND_OF",
#  "connections":[{"character":{"id":"…","name":"Ron Weasley",…},"direction":"outgoing","description":"…"}]}
```

## Prompts

The prompts live in [`app/llm/prompts.py`](app/llm/prompts.py). A test checks that the texts below match it exactly.
Both calls use Structured Outputs. The JSON schema comes from the Pydantic models in
[`app/llm/schemas.py`](app/llm/schemas.py), and their field descriptions also guide the model.

### Character and relation extraction — system

```text
You are an information extraction engine that builds a knowledge graph of characters from articles.

A character is a specific, named person or personified being (human, animal, creature, AI, deity)
that appears in or is discussed by the text. Groups, organisations, places and objects are not characters.

Rules:
1. Extract only what the text states or directly implies. Never use outside knowledge.
2. Merge all mentions of the same character into one entry. Use the most complete proper name as `name`
   and list every other way the text refers to them in `aliases` (first names, surnames, nicknames, titles).
   Do not list pronouns or generic descriptions ("the boy") as aliases.
3. `description` is one or two sentences describing who the character is according to the text.
4. Extract relations only between characters from your `characters` list; `source` and `target` must be
   exactly their `name` values. Never relate a character to itself.
5. `type` is a short verb phrase in UPPER_SNAKE_CASE that reads naturally as `source TYPE target`,
   e.g. FRIEND_OF, PARENT_OF, MARRIED_TO, WORKS_FOR, MENTOR_OF, ENEMY_OF, SIBLING_OF.
   Use the most specific type supported by the text. Prefer a type from the known types list when it
   fits the meaning; invent a new one only when none fits.
6. For directed relations choose the direction carefully (PARENT_OF goes from parent to child).
   For symmetric relations (FRIEND_OF, SIBLING_OF, MARRIED_TO) emit a single relation.
7. If the text contains no characters, return empty lists.
```

### Character and relation extraction — user

```text
Known relation types (reuse when applicable): {known_relation_types}

Article fragment:
<article>
{text}
</article>
```

Response schema (`ExtractionResult`):

```json
{
  "characters": [{"name": "str", "aliases": ["str"], "description": "str"}],
  "relations":  [{"source": "str", "target": "str", "type": "UPPER_SNAKE_CASE", "description": "str"}]
}
```

### Image analysis (vision) — system

```text
You analyse images extracted from documents so that their content can be indexed as text.
Describe only what is visible. Transcribe any readable text, captions, labels and speech verbatim.
For diagrams, charts, family trees and org charts, spell out every depicted entity and every connection
between them explicitly (for example "Arrow from Anna to Boris labelled 'mother of'").
For photos and illustrations, describe the people shown and their interactions, and name them only
when a caption or label identifies them.
Mark the image as not relevant when it is decorative: logos, ornaments, backgrounds, separators,
blank or unreadable content.
```

### Image analysis (vision) — user (followed by the image as a base64 data URL)

```text
This image comes from {context}. Analyse it.
```

Response schema (`ImageAnalysis`): `{"is_relevant": bool, "description": "str"}`.

## Configuration

All settings are environment variables (see [`app/core/config.py`](app/core/config.py) and
[`.env.example`](.env.example)).

| Variable | Default | |
|---|---|---|
| `APP_TOKENS` | — (compose: `local-dev-app-token`) | comma-separated App Tokens |
| `JWT_SECRET` | — (compose: dev value) | ≥ 32 chars |
| `ACCESS_TOKEN_TTL_SECONDS` / `REFRESH_TOKEN_TTL_SECONDS` | 900 / 86400 | |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | — | any OpenAI-compatible backend |
| `LLM_MODEL` / `VISION_MODEL` | `gpt-4o-mini` | must support Structured Outputs; the vision model must accept images |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | `text-embedding-3-small` / 1536 | changing the dimension requires a fresh database (the vector index is created with it) |
| `LLM_CONCURRENCY` | 4 | parallel LLM calls per process |
| `CHUNK_SIZE_CHARS` / `CHUNK_OVERLAP_CHARS` | 12000 / 800 | |
| `PDF_MAX_VISUAL_ITEMS` | 30 | cap on vision calls per document |
| `CHARACTER_MATCH_SCORE` / `CHARACTER_AUTO_MERGE_SCORE` / `RELATION_TYPE_MATCH_SCORE` | 0.93 / 0.97 / 0.95 | Neo4j score scale |

## Tests

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

- **Unit tests** (`tests/unit`): chunking, name and type normalization, in-article merging, PDF parsing
  (text, embedded images, scanned pages, vector graphics, image dedup, cap), the OpenAI client (structured output
  request, refusal handling, base64 images, embedding order), settings, and the README/prompt sync.
- **Integration tests** (`tests/integration`) start a real Neo4j 5.26 via
  [testcontainers](https://testcontainers.com/), so Docker must be running. They go through the HTTP API, with the
  LLM replaced by a deterministic fake (bag-of-words embeddings). They check auth (pair issue, refresh, wrong token
  type, expiry, every protected route), uploads (Markdown text and file, PDF with images, duplicates, validation
  errors, rollback on LLM failure), and every read endpoint. Persistence is checked **directly in Neo4j** with a
  separate driver: nodes, `MENTIONS` and `RELATED_TO` edges, embeddings, and vector-based deduplication of
  characters and relation types.
- To use an already running Neo4j instead: `TEST_NEO4J_URI=bolt://localhost:7687 TEST_NEO4J_PASSWORD=… pytest`.
  **The tests wipe that database.**

## Limitations and next steps

- **Processing is synchronous within the request.** That is enough for a prototype and keeps the API simple, but a
  long PDF can take a minute. For production: return `202` with the article in `processing` status, run the
  pipeline in a worker (Celery/arq/Temporal), and add `GET /articles/{id}`. The status field and idempotency by
  content hash are already in place for this.
- **Concurrent uploads that introduce the same new character** can, rarely, create it twice. This needs a
  per-name lock or a periodic merge job.
- **Entity resolution** relies on thresholds plus a name heuristic. The next step is LLM adjudication for
  borderline candidates (score between the two thresholds).
- **Symmetric relations** (`FRIEND_OF`) are stored in the direction the model emitted them. Queries use
  `direction=any` by default, so this does not affect lookups.
- **Refresh tokens** are not rotated with revocation (stateless JWT). The `jti` claim is there to add a denylist.
- The `text` form field is limited to 1 MB by Starlette's multipart parser; upload larger Markdown as a file.
- Markdown image links are not fetched, which avoids SSRF. Only images embedded in PDFs are analysed.

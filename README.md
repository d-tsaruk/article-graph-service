# Article Graph Service

A FastAPI microservice that reads articles (Markdown or PDF, including their images and graphics), extracts
characters and the relations between them with an LLM, and stores everything in a Neo4j graph. Neo4j's built-in
vector index is used for entity and relation-type deduplication. It is built to be called by a Django monolith.

- **API:** FastAPI, JWT auth (App Token → access + refresh)
- **LLM:** OpenAI or any OpenAI-compatible API, Structured Outputs validated by Pydantic
- **PDF:** PyMuPDF for text, embedded images, scanned pages and vector graphics; a vision model describes the visuals
- **Storage:** Neo4j 5 (graph + native HNSW vector indexes)

## Quick start

```bash
cp .env.example .env          # then set OPENAI_API_KEY in it
docker compose up -d          # or: docker-compose up -d
open http://localhost:8000/docs
```

Without a key every read endpoint works (none of them calls the LLM) and uploads return `503`. Without `.env` the
stack starts with development defaults. Neo4j Browser: http://localhost:7474 (`neo4j` / `local-dev-password`).

## Key decisions and trade-offs

- **One Neo4j for graph and vectors:** one store to run, and a character's edges and embedding commit together.
- **Relation type as an edge property:** the vocabulary is open-ended and Cypher cannot parameterize relationship types.
- **Retrieval plus an LLM judge, not thresholds:** none separates aliases from different people
  ([why](#deduplication-and-the-vector-index)).
- **All LLM calls before one write transaction:** any failure leaves the graph unchanged.
- **Content-hash idempotency, claimed before the LLM work:** a retry of the same bytes never pays twice.
- **Synchronous upload plus `GET /articles/{id}`:** simple now; the next step is `202` plus a worker.
- **Inverse types canonicalized, symmetric types stored once:** one edge per fact.
- **Stateless JWT:** no session store; refresh tokens cannot be revoked.
- **No LLM on reads:** a GET costs nothing, writes nothing and works without an API key.
- **Single tenant, single corpus:** one shared graph; an exact name merges across articles (scoping needs a `corpus` key).

## Architecture

```
client ──JWT──▶ FastAPI api/routes ──▶ pipeline/ingestion.ArticleIngestionService
                  1. documents.parse_pdf / parse_markdown (PyMuPDF), claim: Article{processing}
                  2. LLM vision, 3. chunking, 4. LLM extraction per chunk, 5. merging
                  6. resolution: vector/name candidates → LLM judge (reads and LLM calls only)
                  7. graph/repository: ONE write transaction   ──Bolt──▶ Neo4j 5 (graph + vectors)
                                                               ──HTTPS─▶ OpenAI-compatible API
```

### Ingestion pipeline

1. **Content hash and claim.** SHA-256 of the uploaded bytes. Identical completed content returns the stored result
   (`200`, `"duplicate": true`) with no LLM call. Right after parsing, the upload creates an `Article` in status
   `processing`: this claims the hash. A concurrent retry gets `409` with the `article_id`. A claim older than
   `STALE_PROCESSING_SECONDS` is treated as dead and reprocessed.
2. **Parsing.** Markdown goes to the model as is. A PDF is read per page, and `MAX_PDF_PAGES` is checked first.
   Visuals: embedded images of at least 100 px (each xref once; JPX, JBIG2 and CMYK are re-encoded to PNG), whole
   scanned pages, and pages with many vector drawings (rendered whole). Renders and images are downscaled to
   `PDF_MAX_RENDER_SIDE_PX`, and vision calls are capped by `PDF_MAX_VISUAL_ITEMS`.
3. **Vision.** Each visual goes to the vision model, which returns `{is_relevant, description}`. Relevant
   descriptions are appended to the page text as `[Visual content: …]`, so a relation shown only in a diagram is
   extracted like any other. A visual the model cannot describe is skipped (logged); a backend outage fails the upload
   with `502` so a retry redoes it; if all fail and there is no text: `502`.
4. **Chunking.** Paragraph packing with overlap (`CHUNK_SIZE_CHARS`). A document needing more than `MAX_CHUNKS`
   gets `422`, checked before any LLM call and again after vision.
5. **Extraction.** Chunks run concurrently (`LLM_CONCURRENCY`) through Structured Outputs; refusals, truncation and
   API errors become `502`. Known relation types are passed as preferred types.
6. **In-article merge.** Characters merge only when one normalized name equals the name or an alias of the other.
7. **Resolution against the graph** (see [Deduplication](#deduplication-and-the-vector-index)): reads and LLM calls
   only, except the idempotent seed of the relation-type vocabulary.
8. **Persistence in one transaction.** The article is completed (only while its claim is alive), characters are
   written, then mentions, relation types and relations. Any failure rolls everything back and releases the claim.
   A concurrent upload creating the same new character merges into it (`MERGE` on the unique `name_key`) without
   repeating any paid call. A claim lost to a stale reprocess commits nothing and returns `503` or `409`.

### Deduplication and the vector index

Both indexes are Neo4j native vector indexes (HNSW, cosine); scores use Neo4j's scale `(1 + cos) / 2` ∈ [0, 1].

| Index                     | Node / property          | Embedded text                         | Used for                                      |
|---------------------------|--------------------------|---------------------------------------|-----------------------------------------------|
| `character_embedding`     | `Character.embedding`    | `name (also known as …): description` | candidate characters for the judge            |
| `relation_type_embedding` | `RelationType.embedding` | `friend of` (from `FRIEND_OF`)        | candidate types for the judge, on write       |

**Why not thresholds.** Probes (hand-written pairs, `text-embedding-3-small`) gave Arthur ~ Molly Weasley 0.939,
while true aliases scored 0.78–0.88. In a real run, different characters of the same story reached 0.87, and
relation inverses (bullied by / bullies 0.867) sat inside the synonym range (0.83–0.92). No threshold separates
them, so the index only retrieves candidates and an LLM decides.

**Characters.** An exact canonical-name match merges directly. Otherwise the candidates are the union of stored
characters sharing a name or alias, sharing an identifying name token, and the top `CHARACTER_CANDIDATE_LIMIT` vector
neighbours (no score floor; k is the recall knob). One judge call per upload answers per character: `stored`,
`same_as_new` or `new`. An invalid answer counts as `new`: a duplicate is cheaper than a wrong merge.
A deterministic guard backs the judge: two characters whose full names share a surname but not a given name
(`Percy Weasley` / `Ron Weasley`) are never offered to it as a pair. Each merge group the verdicts form is then
checked member by member against the names already accepted into it (the stored character's, or the canonical
member's), so a chain such as `Percy` -> `Ronald` -> `Ron Weasley` cannot bring a conflicting member in: it is kept as
its own character. A given name that prefixes the other (`Ron` / `Ronald`) is not a conflict. The live judge did merge
siblings without it.

**Relation types.** The canonical vocabulary is seeded on first use. An exact match is used directly; otherwise the
top 5 neighbours with `score ≥ RELATION_TYPE_CANDIDATE_SCORE` go to one judge call, which maps only to the same or a
more general type **in the same direction** (`FATHER_OF` → `PARENT_OF`; the nuance stays in the edge description).
Only types used by a stored edge get a node.

**Inverses.** An inverse type from the extraction prompt (`Ron CHILD_OF Arthur`) is converted before the judge by
swapping the ends, so none is stored, and an inverse is never offered to the judge as a candidate. Symmetric types
(`FRIEND_OF`, `SIBLING_OF`, …) are stored once per pair, in the first-seen orientation.

**Query.** `GET /relations/connected/{id}?type=…` resolves the type as: known inverse (`CHILD_OF` → `PARENT_OF` with
the direction flipped) → exact stored or canonical name → the name with `_OF` appended (`friend` → `FRIEND_OF`) →
`resolved_type: null`. It calls no LLM and writes nothing, so query with a name from `/relations/types`. For a
symmetric type `direction` is ignored and every connection is reported as `outgoing`.

**Cost per upload.** One extraction call per chunk, one vision call per visual (at most `PDF_MAX_VISUAL_ITEMS`,
`detail: high`), 2-3 embedding calls, at most 2 judge calls. Queries are free.

### Graph schema

```
(:Article {id, title, source_type, content_hash, text, status, created_at, completed_at})
    -[:MENTIONS]->
(:Character {id, name, name_key, aliases[], alias_keys[], description, embedding[], created_at})
    -[:RELATED_TO {type, description, article_ids[], created_at}]->
(:Character)

(:RelationType {name, embedding[], created_at})      // vocabulary of relation types
```

| Element                  | Constraint / index                                          |
|--------------------------|-------------------------------------------------------------|
| `Article.id`, `Character.id`, `RelationType.name` | unique                       |
| `Article.content_hash`   | unique (idempotent uploads; the claim of a hash)            |
| `Character.name_key`     | **unique** (an exact name is one character, even under concurrent uploads) |
| `Character.embedding`    | vector index `character_embedding`                          |
| `RelationType.embedding` | vector index `relation_type_embedding`                      |
| `RELATED_TO.type`        | relationship property index                                 |

The unique `name_key` constraint replaces an earlier index. A database that already holds duplicate keys fails at
startup: re-ingest into a fresh volume. Relations are directed (`PARENT_OF` goes from parent to child); one edge per
`(source, type, target)` is kept and `article_ids` lists the articles where it was found. IDs are UUIDs.

## API

| Method | Path                                         | Auth    | Description                                         |
|--------|----------------------------------------------|---------|-----------------------------------------------------|
| POST   | `/auth/token`                                | —       | App Token → `{access_token, refresh_token}`         |
| POST   | `/auth/refresh`                              | —       | refresh token → new pair                            |
| POST   | `/articles/upload`                           | Bearer  | multipart: `text` (Markdown) **or** `file` (.md/.pdf), optional `title` |
| GET    | `/articles/{article_id}`                     | Bearer  | status (`processing`, `stale`, `completed`) and, once completed, characters and relations |
| GET    | `/characters?limit=&offset=`                 | Bearer  | all characters, sorted by name                      |
| GET    | `/characters/by_article/{article_id}`        | Bearer  | characters mentioned in an article (`404` if unknown) |
| GET    | `/relations/types`                           | Bearer  | unique relation types with counts                   |
| GET    | `/relations/connected/{character_id}?type=&direction=` | Bearer | characters connected by a relation type |
| GET    | `/health`                                    | —       | liveness + Neo4j connectivity (`503` when Neo4j is down) |

Interactive docs: http://localhost:8000/docs. `/health`, `/docs` and `/openapi.json` are public on purpose.
Error codes: `409` identical content is being processed (the body carries `article_id`); `413` upload too large
(`MAX_UPLOAD_BYTES`); `415` unsupported file type; `422` unreadable or empty document, or over the page
(`MAX_PDF_PAGES`) or chunk (`MAX_CHUNKS`) limit; `502` LLM backend error (generic message, details in the logs);
`503` LLM not configured, or the upload was superseded by a stale reprocess (retry).

### Authentication

App Tokens (`APP_TOKENS`, comma-separated) represent client applications such as the Django monolith. A client
exchanges its App Token for an **access** token (15 min) and a **refresh** token (24 h). Both are HMAC-signed JWTs
with a `type` claim, so neither can be used in place of the other. They carry a `jti` for a future denylist.

### curl examples

```bash
BASE=http://localhost:8000

# 1. Exchange the App Token for a token pair
TOKENS=$(curl -s -X POST $BASE/auth/token -H 'Content-Type: application/json' \
  -d '{"app_token": "local-dev-app-token"}')
ACCESS=$(echo "$TOKENS" | python3 -c 'import sys, json; print(json.load(sys.stdin)["access_token"])')
REFRESH=$(echo "$TOKENS" | python3 -c 'import sys, json; print(json.load(sys.stdin)["refresh_token"])')
AUTH="Authorization: Bearer $ACCESS"

# 2. Refresh when the access token expires
curl -s -X POST $BASE/auth/refresh -H 'Content-Type: application/json' -d "{\"refresh_token\": \"$REFRESH\"}"

# 3a. Upload Markdown text
curl -s -X POST $BASE/articles/upload -H "$AUTH" \
  -F 'text=# The Boy Who Lived

Harry Potter lived with his aunt Petunia Dursley and uncle Vernon Dursley.
On the Hogwarts Express Harry met Ron Weasley, who became his best friend.'

# 3b. Upload a Markdown or PDF file (title is optional)
curl -s -X POST $BASE/articles/upload -H "$AUTH" -F 'file=@book.pdf' -F 'title=Chapter 1'
# {"article_id":"6f1c…","title":"The Boy Who Lived","source_type":"markdown","duplicate":false,
#  "characters":[{"id":"…","name":"Harry Potter","aliases":["Harry"],"description":"…"}, …],
#  "relations":[{"source_name":"Harry Potter","target_name":"Ron Weasley","type":"FRIEND_OF","description":"…"}, …]}

# 4. Article status and result; all characters; characters of an article; relation types
curl -s $BASE/articles/<article_id> -H "$AUTH"
curl -s $BASE/characters -H "$AUTH"
curl -s $BASE/characters/by_article/<article_id> -H "$AUTH"
curl -s $BASE/relations/types -H "$AUTH"

# 5. Who is a friend of Harry? `type` accepts FRIEND_OF, "friend of" or "friend"
HARRY=$(curl -s "$BASE/characters?limit=1000" -H "$AUTH" | python3 -c \
  'import sys, json; print(next(c["id"] for c in json.load(sys.stdin) if c["name"] == "Harry Potter"))')
curl -s -G $BASE/relations/connected/$HARRY -H "$AUTH" --data-urlencode 'type=friend of'
curl -s -G $BASE/relations/connected/$HARRY -H "$AUTH" -d type=PARENT_OF -d direction=incoming
# {"character_id":"…","requested_type":"friend of","resolved_type":"FRIEND_OF",
#  "connections":[{"character":{"id":"…","name":"Ron Weasley",…},"direction":"outgoing","description":"…"}]}
```

## Calling from Django

- Call from a Celery task, not from a web worker (a PDF upload can take minutes), and store the returned `article_id`.
- Retrying the same bytes is safe: `201` means new, `200` with `duplicate: true` means already done, and `409` with an
  `article_id` means another request is in progress.
- On `409`, poll `GET /articles/{id}`: `completed` means done; `stale` or `404` means that run died, so upload again.
- Retry with backoff on `502` and on a `503` "superseded". A `503` "not configured", `413`, `415` and `422` are
  permanent. Refresh the access token on `401`.

```python
token = requests.post(f"{BASE}/auth/token", json={"app_token": APP_TOKEN}, timeout=10).json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
requests.post(f"{BASE}/articles/upload", files={"file": ("a.pdf", pdf_bytes)}, headers=headers, timeout=600)
```

## Sample articles

- `samples/01-two-families.md` and `samples/02-summer-at-the-burrow.md` exercise dedup: upload them in order. Expected
  with a capable model: one node each for Ron / Ronald, Harry / the Boy Who Lived and Voldemort / Tom Riddle; separate
  Arthur, Molly and Percy; family relations as `PARENT_OF` / `MARRIED_TO` / `RELATIVE_OF`, and no `*_BY` types.
- `samples/03-bakery-chart.pdf` exercises the vision path: its relations appear only in the embedded chart.
  Expected edges: `Elena Varga PARENT_OF Mateo Rossi` and `Ivo Petrov WORKS_FOR Elena Varga` (the text states neither).
  Upload with `-F 'file=@samples/03-bakery-chart.pdf'`; regenerate with `.venv/bin/python scripts/make_sample_pdf.py`.

## Prompts

The prompts live in [`app/llm/prompts.py`](app/llm/prompts.py). A test checks that the texts below match it exactly.
All calls use Structured Outputs. The JSON schema comes from the Pydantic models in
[`app/llm/schemas.py`](app/llm/schemas.py), and their field descriptions also guide the model.

### Character and relation extraction — system

```text
You are an information extraction engine that builds a knowledge graph of characters from articles.

A character is a specific, named person or personified being (human, animal, creature, AI, deity)
that appears in or is discussed by the text. Groups, organisations, places and objects are not characters.

Rules:
1. Extract only what the text states or directly implies. Never use outside knowledge.
2. Merge all mentions of the same character into one entry. Use the most complete proper name as `name`
   and list every other way the text refers to them in `aliases` (first names, nicknames, titles with surnames).
   Do not list pronouns or generic descriptions ("the boy") as aliases. Never list an alias that the text
   also uses for another character, such as a bare family surname.
   An alias is another name for the same character, never a description of a relation such as "Ronald's father"
   or "Harry's aunt"; such phrases belong to other characters or to relations.
3. `description` is one or two sentences describing who the character is according to the text.
4. Extract relations only between characters from your `characters` list; `source` and `target` must be
   exactly their `name` values. Never relate a character to itself.
5. `type` is a relation type in UPPER_SNAKE_CASE that reads naturally as `source TYPE target`.
   Use one of these canonical types whenever it fits, choosing the general type over a specific one:
   - family: PARENT_OF, SIBLING_OF, MARRIED_TO, RELATIVE_OF (aunts, uncles, cousins, grandparents, in-laws)
   - care: GUARDIAN_OF
   - social: FRIEND_OF, ENEMY_OF, ROMANTIC_PARTNER_OF
   - work: WORKS_FOR, TEACHER_OF, MENTOR_OF
   - actions: BULLIES, KILLED, PROTECTS
   Otherwise prefer a type from the known types list, and invent a new general type only when nothing fits.
   Put the nuance into the relation `description`, not into the type: "father of" is PARENT_OF,
   "best friends" is FRIEND_OF, "uncle of" is RELATIVE_OF.
6. Direction: `source` is the one who holds the role or performs the action: PARENT_OF goes from
   parent to child, GUARDIAN_OF from guardian to ward, WORKS_FOR from employee to employer,
   KILLED from killer to victim, BULLIES from bully to victim. Text is often written from the
   other side; then either reorder the pair or use an inverse type, which is converted automatically:
   CHILD_OF, WARD_OF, EMPLOYS, STUDENT_OF, MENTEE_OF, BULLIED_BY, KILLED_BY, PROTECTED_BY.
   Examples: "Ronald's father Arthur" -> Arthur PARENT_OF Ronald (or Ronald CHILD_OF Arthur);
   "his parents were killed by Voldemort" -> Voldemort KILLED <each parent>;
   "raised by his aunt Petunia" -> Petunia GUARDIAN_OF Harry.
   For symmetric types (SIBLING_OF, MARRIED_TO, RELATIVE_OF, FRIEND_OF, ENEMY_OF, ROMANTIC_PARTNER_OF)
   emit a single relation.
7. Extract every family relation the text states, even in passing: parents and children, spouses, siblings,
   aunts, uncles and cousins, guardians.
   Cousins, aunts, uncles, grandparents and in-laws are RELATIVE_OF, never SIBLING_OF; SIBLING_OF is only for
   brothers and sisters.
8. Split compound facts into basic relations: "Voldemort killed Harry's parents" becomes KILLED from
   Voldemort to each parent who is a character.
9. Use WORKS_FOR for a work hierarchy (a teacher who reports to a headmaster WORKS_FOR the headmaster).
   Use MENTOR_OF only when the text describes personal guidance.
10. If the text contains no characters, return empty lists.
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

### Character judge — system

```text
You decide whether newly extracted characters are the same individuals as characters already stored
in a knowledge graph, or as each other.

You receive "Known characters" (references S1, S2, ...) and "New characters to resolve"
(references N1, N2, ...). Each new character lists the known characters that are its candidates.

For every new character return exactly one verdict:
- decision "stored", match = one of ITS candidate references, when it is certainly the same individual
  as that known character;
- decision "same_as_new", match = another N reference, when it is the same individual as another new
  character and matches none of its candidates;
- decision "new", match = null, otherwise.

Rules:
1. Same individual means the same person or being, not merely the same family, role or name part.
   Family members who share a surname (parent and child, husband and wife, siblings) are different individuals.
2. A first name, a nickname, a title with a surname, a full name, an epithet or a former or secret name can
   refer to the same individual when the descriptions are consistent with each other.
3. Contradicting descriptions (different parents, spouses, ages or roles) mean different individuals.
4. If you are not sure, answer "new". A missed merge is cheaper than a wrong one.
5. Use only the information given. Write the `reason` before deciding.
```

### Character judge — user

```text
Known characters:
{known}

New characters to resolve:
{new}
```

### Relation-type judge — system

```text
You keep the vocabulary of relation types in a knowledge graph free of synonyms.
A relation type reads as `source TYPE target`, for example `Arthur PARENT_OF Ron`.

You receive "Known relation types" (references S1, S2, ...) and "New relation types to resolve"
(references N1, N2, ...). Each new type lists the known types that are its candidates.

For every new type return exactly one verdict:
- decision "stored", match = one of ITS candidate references, when that known type means the same
  as the new type, or is a more general form of it, in the same direction;
- decision "same_as_new", match = another N reference, under the same condition, when none of its
  candidates fits; point to the more general of the two types, and if several qualify, to the one
  closest in meaning;
- decision "new", match = null, otherwise.

Rules:
1. Direction matters. Map only if `A NEW_TYPE B` implies `A KNOWN_TYPE B` for the same A and B.
   An inverse is never the same: CHILD_OF is not PARENT_OF, BULLIED_BY is not BULLIES,
   EMPLOYS is not WORKS_FOR.
2. Different or opposite relations are never the same: DIVORCED_FROM is not MARRIED_TO,
   ENEMY_OF is not FRIEND_OF.
3. Map a specific type to its general form: FATHER_OF -> PARENT_OF, SPOUSE_OF -> MARRIED_TO,
   BEST_FRIENDS_WITH -> FRIEND_OF. Never map a general type to a more specific one: PARENT_OF is not FATHER_OF.
4. If several known types qualify, choose the most specific of them, the one closest in meaning:
   FATHER_OF -> PARENT_OF, not RELATIVE_OF.
5. If you are not sure, answer "new".
6. Write the `reason` before deciding.
```

### Relation-type judge — user

```text
Known relation types:
{known}

New relation types to resolve:
{new}
```

Both judges answer with `Judgement` (`app/llm/schemas.py`):

```json
{"verdicts": [{"item": "N1", "reason": "str", "decision": "stored|same_as_new|new", "match": "S1|N2|null"}]}
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
| `JWT_ALGORITHM` | `HS256` | `HS256`, `HS384` or `HS512` |
| `ACCESS_TOKEN_TTL_SECONDS` / `REFRESH_TOKEN_TTL_SECONDS` | 900 / 86400 | |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | — | any OpenAI-compatible backend |
| `NEO4J_DATABASE` | `neo4j` | |
| `LLM_MODEL` | `gpt-4.1-mini` | extraction model; must support Structured Outputs |
| `VISION_MODEL` | `gpt-4o-mini` | must accept images |
| `JUDGE_MODEL` | `gpt-4o-mini` | must support Structured Outputs |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | `text-embedding-3-small` / 1536 | changing the dimension requires a fresh database (the vector index is created with it) |
| `LLM_CONCURRENCY` | 4 | parallel LLM calls per process |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` | 120 / 3 | per LLM call |
| `MAX_UPLOAD_BYTES` | 20971520 | larger uploads get 413 |
| `MAX_PDF_PAGES` / `MAX_CHUNKS` | 100 / 50 | more pages or chunks get 422 (chunks are checked before any LLM call) |
| `STALE_PROCESSING_SECONDS` | 1800 | a `processing` claim older than this is treated as dead and reprocessed |
| `CHUNK_SIZE_CHARS` / `CHUNK_OVERLAP_CHARS` | 12000 / 800 | |
| `PDF_MAX_VISUAL_ITEMS` | 30 | cap on vision calls per document |
| `PDF_MIN_IMAGE_SIDE_PX` / `PDF_RENDER_DPI` | 100 / 150 | smaller embedded images are ignored; DPI of page renders |
| `PDF_MAX_RENDER_SIDE_PX` | 2048 | renders and embedded images are downscaled to this side |
| `PDF_SCANNED_PAGE_MAX_CHARS` / `PDF_DRAWINGS_RENDER_THRESHOLD` | 50 / 30 | a page with fewer characters counts as scanned; one with more drawings is rendered whole |
| `CHARACTER_CANDIDATE_LIMIT` | 10 | vector candidates per new character (recall knob) |
| `RELATION_TYPE_CANDIDATE_SCORE` | 0.75 | Neo4j score scale |
| `API_PORT` / `NEO4J_HTTP_PORT` / `NEO4J_BOLT_PORT` | 8000 / 7474 / 7687 | host ports (compose) |

Every variable is passed through `docker-compose.yml`. The app logs a warning at startup while the dev `JWT_SECRET` or
`APP_TOKENS` is in use.

Stronger models (e.g. `gpt-4o`) for `LLM_MODEL` and `JUDGE_MODEL` improve extraction and dedup quality at a higher cost.

**Model choice.** `LLM_MODEL` defaults to `gpt-4.1-mini`. In an experiment of 2 runs per model over the two sample
articles, `gpt-4o-mini` inverted 4 to 6 relation edges (for example child and parent swapped) and `gpt-4.1-mini`
inverted none, at similar token counts. This is a small sample, so treat it as an indication, not a benchmark.

## Tests

Python 3.12+ and Docker are required.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pytest
```

- **Unit tests** (`tests/unit`): chunking, normalization, in-article merging, PDF parsing and its limits, the OpenAI
  client, settings, compose/`.env.example`/README completeness, verdict grouping, prompt and README sync, and the
  generated sample PDF.
- **Integration tests** (`tests/integration`) start a real Neo4j 5.26 via [testcontainers](https://testcontainers.com/)
  and go through the HTTP API, with the LLM replaced by a deterministic fake. They check auth, uploads (PDF with images,
  duplicates, in-progress claims, stale reclaim, lost claim, concurrent creation of one character, rollback of a failed
  write, failed visuals), `GET /articles/{id}`, every read endpoint (no LLM call on a query) and the schema migration.
  Persistence is checked **directly in Neo4j**; the fake judge proves the decision path, not model quality.
- To use a running Neo4j: `TEST_NEO4J_URI=bolt://localhost:7687 TEST_NEO4J_PASSWORD=… pytest`. **The tests wipe it.**

## Limitations and next steps

- **Upload is synchronous.** For production return `202`, run the pipeline in a worker (Celery/arq/Temporal) and poll
  `GET /articles/{id}`; today a client polls it after a `409`. The claim and the content hash are already in place.
- **Recall is bounded by the candidate channels.** An alias with no shared name token that ranks outside the vector
  top-k is never judged, and becomes a duplicate.
- **The extraction prompt's examples use the sample cast**: neutral ones regressed dedup on the live run, so they stay.
- **One relation per pair from the extraction model.** gpt-4.1-mini tends to keep only the most salient relation of
  a pair: `Dudley BULLIES Harry` is extracted, `Dudley RELATIVE_OF Harry` (cousins) often is not. Asking for both in
  the prompt fixed it on the live run but traded it for other errors (a false `KILLED` edge), so the prompt stays.
- **Edges vary between runs** even at `temperature=0`: a live run may add one wrong edge (`Arthur WORKS_FOR Ron` from
  "Ronald's father, who works at the Ministry"). Dedup is backed by code; edge extraction is not.
- **Same surname, different given name never merges**, so `Bill Weasley` / `William Weasley` stay apart.
- **Existing data is not re-resolved.** Earlier wrong merges or types stay until re-ingested into a fresh database.
- **A merged character's embedding is not recomputed.** Concurrent merges into one character can overwrite aliases.
- **In-article merging** is name-anchored and does not use the judge; a relation that names an ambiguous alias is
  dropped rather than guessed. Possessive aliases ("Ronald's father") are dropped.
- **Queries resolve a type only through the inverse map, the exact name or the `_OF` suffix.** A paraphrase such as
  `son of` resolves to nothing: no LLM runs on a GET. `direction` is ignored for symmetric types (`FRIEND_OF`).
- **The upload size is checked after the body is received**, so put a body limit on the reverse proxy.
- The `text` form field is limited to 1 MB by Starlette; upload larger Markdown as a file.

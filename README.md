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
                   │                   ├─ 6. resolution: vector/name candidates → LLM judge │
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
   response is validated before the pipeline sees it. Refusals, truncation and API errors become `502`. The
   canonical relation vocabulary and the types already used in the graph are passed as preferred types.
6. **In-article merge.** Characters from all chunks are unified with union-find over normalized names and aliases.
   Relations are rewritten to canonical names, typed (`UPPER_SNAKE_CASE`) and deduplicated. Relations that point to
   unknown characters, and self-loops, are dropped.
7. **Resolution against the graph** (see [Deduplication](#deduplication-and-the-vector-index)). Only reads and LLM
   calls, no writes, except the idempotent seed of the relation-type vocabulary.
8. **Persistence.** The article is created as `processing`. Characters, mentions and relations are written, then
   the article is marked `completed`. Every LLM call, including both judges, finishes before the first write, so an
   LLM failure (502) leaves the graph unchanged except for the idempotent vocabulary seed. A Neo4j failure during
   the writes deletes the article and its relations. Characters created or updated and relation types created
   before the failure remain.

### Deduplication and the vector index

Both indexes are Neo4j native vector indexes (HNSW, cosine). Neo4j reports cosine similarity as
`score = (1 + cos) / 2` ∈ [0, 1]; all scores below are on that scale.

| Index                     | Node / property          | Embedded text                         | Used for                                               |
|---------------------------|--------------------------|---------------------------------------|--------------------------------------------------------|
| `character_embedding`     | `Character.embedding`    | `name (also known as …): description` | candidate characters for the judge                     |
| `relation_type_embedding` | `RelationType.embedding` | `friend of` (from `FRIEND_OF`)        | candidate types for the judge, on write and on query   |

**Why not thresholds.** Probes (hand-written pairs, `text-embedding-3-small`) gave Arthur ~ Molly Weasley 0.939,
while true aliases scored 0.78–0.88. In a real run, different characters of the same story reached 0.87, and
relation inverses (bullied by / bullies 0.867) sat inside the synonym range (0.83–0.92). No threshold separates
them, so the index only retrieves candidates and an LLM decides.

**Characters.**

1. An exact canonical-name match merges directly.
2. Otherwise the candidates are the union of three channels: stored characters sharing a name or alias (`Weasley`);
   stored characters sharing an identifying name token (honorifics are ignored); and the top
   `CHARACTER_CANDIDATE_LIMIT` (10) vector neighbours, with no score floor. k is the recall knob.
3. One judge call per upload sees all candidates and all new characters (references `S…` / `N…`), and answers per
   character: `stored`, `same_as_new` or `new`. In-article duplicates from different chunks are caught there.
4. Invalid answers (an unknown reference, a candidate that was not offered, a missing item) count as `new`: a
   duplicate is cheaper than a wrong merge.

**Relation types.**

- The canonical vocabulary (`PARENT_OF`, `SIBLING_OF`, `MARRIED_TO`, `RELATIVE_OF`, `GUARDIAN_OF`, `FRIEND_OF`,
  `ENEMY_OF`, `ROMANTIC_PARTNER_OF`, `WORKS_FOR`, `TEACHER_OF`, `MENTOR_OF`, `BULLIES`, `KILLED`, `PROTECTS`) is seeded
  as `RelationType` nodes on first use.
- An exact match is used directly.
- Otherwise the top 5 neighbours with `score ≥ RELATION_TYPE_CANDIDATE_SCORE` (0.75) go to one judge call per upload.
  The judge maps only to the same or a more general type **in the same direction**: inverses and opposites never
  merge. A specific type collapses to the general one (`FATHER_OF` → `PARENT_OF`), and the nuance stays in the
  relation `description`. The known inverse of a type is never offered as its candidate, so this holds in code and
  not only in the prompt.
- Only types used by stored edges get a node.
- **Direction at extraction.** The extraction prompt may emit an inverse type (`Ron CHILD_OF Arthur`). Code converts it
  deterministically before the judge, by swapping source and target (`Arthur PARENT_OF Ron`), so stored data is always
  canonical and no inverse type is ever stored. Symmetric types (`FRIEND_OF`, `SIBLING_OF`, ...) are stored once per
  pair, within an article and across articles, in the first-seen orientation. The inverse rule still holds for stored
  data and the judge: stored edges are never re-oriented and an inverse is never merged with its counterpart.

**Query.** `GET /relations/connected/{id}?type=…` resolves the type in this order:

1. a known inverse, checked first so that a stray stored `CHILD_OF` cannot hide the `PARENT_OF` edges
   (`CHILD_OF` → `PARENT_OF`, `WARD_OF` → `GUARDIAN_OF`, `EMPLOYS` → `WORKS_FOR`,
   `STUDENT_OF` → `TEACHER_OF`, `MENTEE_OF` → `MENTOR_OF`, `BULLIED_BY` → `BULLIES`, `KILLED_BY` → `KILLED`,
   `PROTECTED_BY` → `PROTECTS`), answered with the direction flipped and no LLM call;
2. exact match;
3. vector candidates, then the judge;
4. otherwise `resolved_type: null`.

**Cost.** At most 2 extra small chat calls per upload (`gpt-4o-mini` by default; token usage is logged at INFO).
A query costs one call only for a non-exact, non-inverse type.

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

## Sample articles

`samples/01-two-families.md` and `samples/02-summer-at-the-burrow.md` exercise dedup end to end. Upload them in
order with the `3b` command above (`-F 'file=@samples/01-two-families.md'`, then the second file), then inspect
`GET /characters` and `GET /relations/types`.

Expected outcome with a capable model:

- one node each for Ron / Ronald, Harry / the Boy Who Lived and Voldemort / Tom Riddle;
- separate Arthur, Molly and Percy;
- family relations as `PARENT_OF` / `MARRIED_TO` / `RELATIVE_OF`, and no `*_BY` types.

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
| `ACCESS_TOKEN_TTL_SECONDS` / `REFRESH_TOKEN_TTL_SECONDS` | 900 / 86400 | |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | — | any OpenAI-compatible backend |
| `LLM_MODEL` / `VISION_MODEL` | `gpt-4o-mini` | must support Structured Outputs; the vision model must accept images |
| `JUDGE_MODEL` | `gpt-4o-mini` | must support Structured Outputs |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | `text-embedding-3-small` / 1536 | changing the dimension requires a fresh database (the vector index is created with it) |
| `LLM_CONCURRENCY` | 4 | parallel LLM calls per process |
| `CHUNK_SIZE_CHARS` / `CHUNK_OVERLAP_CHARS` | 12000 / 800 | |
| `PDF_MAX_VISUAL_ITEMS` | 30 | cap on vision calls per document |
| `CHARACTER_CANDIDATE_LIMIT` | 10 | vector candidates per new character (recall knob) |
| `RELATION_TYPE_CANDIDATE_SCORE` | 0.75 | Neo4j score scale |
| `API_PORT` / `NEO4J_HTTP_PORT` / `NEO4J_BOLT_PORT` | 8000 / 7474 / 7687 | host ports (compose) |

Stronger models (e.g. `gpt-4o`) for `LLM_MODEL` and `JUDGE_MODEL` improve extraction and dedup quality at a higher cost.

## Tests

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

- **Unit tests** (`tests/unit`): chunking, name and type normalization, in-article merging, PDF parsing
  (text, embedded images, scanned pages, vector graphics, image dedup, cap), the OpenAI client (structured output
  request, refusal handling, base64 images, embedding order), settings, judge verdict grouping, vocabulary and
  prompt consistency, and the README/prompt sync.
- **Integration tests** (`tests/integration`) start a real Neo4j 5.26 via
  [testcontainers](https://testcontainers.com/), so Docker must be running. They go through the HTTP API, with the
  LLM replaced by a deterministic fake (bag-of-words embeddings). They check auth (pair issue, refresh, wrong token
  type, expiry, every protected route), uploads (Markdown text and file, PDF with images, duplicates, validation
  errors, rollback on LLM failure), and every read endpoint. Persistence is checked **directly in Neo4j** with a
  separate driver: nodes, `MENTIONS` and `RELATED_TO` edges, embeddings, candidate retrieval and LLM-judge dedup of
  characters and relation types (a deterministic fake judge declares which names are the same, so the tests prove the
  decision path, not model quality), inverse queries, and rollback on judge failure.
- To use an already running Neo4j instead: `TEST_NEO4J_URI=bolt://localhost:7687 TEST_NEO4J_PASSWORD=… pytest`.
  **The tests wipe that database.**

## Limitations and next steps

- **Processing is synchronous within the request.** That is enough for a prototype and keeps the API simple, but a
  long PDF can take a minute. For production: return `202` with the article in `processing` status, run the
  pipeline in a worker (Celery/arq/Temporal), and add `GET /articles/{id}`. The status field and idempotency by
  content hash are already in place for this.
- **Concurrent uploads that introduce the same new character** can, rarely, create it twice. This needs a
  per-name lock or a periodic merge job.
- **Recall is bounded by the candidate channels.** An alias with no shared name token that ranks outside the vector
  top-k is never judged, and becomes a duplicate.
- **Existing data is not re-resolved.** Earlier wrong merges or types stay until re-ingested into a fresh database.
- **In-article merging** (union-find over names and aliases) does not use the judge. The prompt forbids shared-surname
  aliases, but a model that ignores this merges two people.
- **Only the listed inverses are answered on query.** A paraphrased inverse such as `son of` resolves to nothing.
- **One judge call per upload is not split.** Very large articles produce one large prompt.
- **Symmetric relations** (`FRIEND_OF`) are stored once, in the direction first emitted. Queries use
  `direction=any` by default, so this does not affect lookups.
- **Refresh tokens** are not rotated with revocation (stateless JWT). The `jti` claim is there to add a denylist.
- The `text` form field is limited to 1 MB by Starlette's multipart parser; upload larger Markdown as a file.
- Markdown image links are not fetched, which avoids SSRF. Only images embedded in PDFs are analysed.

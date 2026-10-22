import asyncio
import hashlib
import logging
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePath

from app.core.config import Settings
from app.graph.repository import (
    ArticleRecord,
    ArticleStatus,
    CharacterRecord,
    ClaimLostError,
    DuplicateArticleError,
    GraphRepository,
    GraphWriter,
    RelationRecord,
)
from app.llm.base import LLMClient, LLMError, LLMNotConfiguredError, LLMUnavailableError
from app.llm.schemas import ExtractionResult, ImageAnalysis
from app.llm.vocabulary import CANONICAL_RELATION_TYPES, SYMMETRIC_RELATION_TYPES
from app.pipeline.chunking import chunk_text
from app.pipeline.documents import DocumentError, Page, ParsedDocument, PdfOptions, Visual, parse_markdown, parse_pdf
from app.pipeline.merging import merge_extractions
from app.pipeline.resolution import CharacterResolver, RelationTypeResolver, TypePlan

logger = logging.getLogger(__name__)

KNOWN_RELATION_TYPES_IN_PROMPT = 100
VISION_CONTEXT_TEXT_CHARS = 500


class SourceType(StrEnum):
    MARKDOWN = "markdown"
    PDF = "pdf"


class ArticleInProgressError(Exception):
    def __init__(self, article_id: str) -> None:
        super().__init__(article_id)
        self.article_id = article_id


@dataclass(frozen=True)
class UploadedDocument:
    content: bytes
    source_type: SourceType
    title: str | None = None
    filename: str | None = None


@dataclass(frozen=True)
class IngestionResult:
    article_id: str
    title: str
    source_type: str
    duplicate: bool
    characters: list[CharacterRecord]
    relations: list[RelationRecord]


def unwrap_task_failure(failure: ExceptionGroup) -> BaseException:
    """Logs every failed task and picks the one to raise: an LLMError (502/503) if any, else the first."""
    for exc in failure.exceptions:
        logger.error("Task failed: %r", exc, exc_info=exc)
    return next((exc for exc in failure.exceptions if isinstance(exc, LLMError)), failure.exceptions[0])


async def _run_all[T](coros: list[Coroutine[object, object, T]]) -> list[T]:
    """Runs the coroutines concurrently and returns their results in order. The first failure cancels the siblings."""
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(coro) for coro in coros]
    except ExceptionGroup as failure:
        # Plain `except`, not `except*`: raising inside `except*` would wrap the error in a group again.
        raise unwrap_task_failure(failure) from None
    return [task.result() for task in tasks]


class ArticleIngestionService:
    """hash lookup -> parse -> claim the hash (`processing` article) -> describe visuals -> chunk -> extract -> merge
    -> plan resolution (reads + LLM judges) -> one write transaction. The claim makes a concurrent retry of the
    same bytes fail fast with 409; every LLM call finishes before the transaction, and a failure releases the claim."""

    def __init__(
        self,
        repository: GraphRepository,
        llm: LLMClient,
        settings: Settings,
        relation_types: RelationTypeResolver,
    ) -> None:
        self._repository = repository
        self._llm = llm
        self._settings = settings
        self._relation_types = relation_types
        self._characters = CharacterResolver(repository, llm, candidate_limit=settings.character_candidate_limit)
        self._pdf_options = PdfOptions.from_settings(settings)

    async def ingest(self, upload: UploadedDocument) -> IngestionResult:
        content_hash = hashlib.sha256(upload.content).hexdigest()
        existing = await self._repository.find_article_by_hash(content_hash)
        if existing is not None:
            if existing.status is ArticleStatus.COMPLETED:
                return await self._existing_result(existing)
            if not existing.is_stale(self._settings.stale_processing_seconds):
                raise ArticleInProgressError(existing.id)
            logger.warning("Reprocessing stale article %s", existing.id)
            await self._repository.delete_unfinished_article(
                existing.id, older_than_seconds=self._settings.stale_processing_seconds
            )

        document = await self._parse(upload)
        self._chunks("\n\n".join(page.text for page in document.pages))
        title = upload.title or document.title or (PurePath(upload.filename).stem if upload.filename else None)

        article_id = str(uuid.uuid4())
        try:
            await self._repository.create_article(
                article_id=article_id,
                title=title or "Untitled",
                source_type=upload.source_type.value,
                content_hash=content_hash,
            )
        except DuplicateArticleError:
            return await self._settled(content_hash)

        try:
            text = await self._build_text(document, title or "untitled")
            if not text.strip():
                raise DocumentError("no text or relevant visual content could be extracted")
            extraction = await self._extract(text)
            return await self._persist(
                article_id=article_id,
                title=title or "Untitled",
                source_type=upload.source_type,
                text=text,
                extraction=extraction,
            )
        except BaseException as exc:
            if not isinstance(exc, ClaimLostError):
                logger.error("Ingesting article %s failed, releasing its claim", article_id)
            try:
                await self._repository.delete_unfinished_article(article_id)
            except Exception:
                logger.exception("Releasing the claim of article %s failed", article_id)
            if isinstance(exc, ClaimLostError):
                return await self._settled(content_hash)
            raise

    async def _settled(self, content_hash: str) -> IngestionResult:
        """Another upload of the same bytes won the claim: report what it is doing or what it finished."""
        existing = await self._repository.find_article_by_hash(content_hash)
        if existing is None:
            raise ClaimLostError(content_hash)
        if existing.status is ArticleStatus.COMPLETED:
            return await self._existing_result(existing)
        raise ArticleInProgressError(existing.id)

    async def _existing_result(self, article: ArticleRecord) -> IngestionResult:
        characters = await self._repository.get_article_characters(article.id) or []
        relations = await self._repository.get_article_relations(article.id)
        return IngestionResult(
            article_id=article.id,
            title=article.title,
            source_type=article.source_type,
            duplicate=True,
            characters=characters,
            relations=relations,
        )

    async def _parse(self, upload: UploadedDocument) -> ParsedDocument:
        if upload.source_type is SourceType.PDF:
            return await asyncio.to_thread(parse_pdf, upload.content, self._pdf_options)
        try:
            return parse_markdown(upload.content.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise DocumentError("markdown must be UTF-8 encoded") from exc

    def _chunks(self, text: str) -> list[str]:
        chunks = chunk_text(text, self._settings.chunk_size_chars, self._settings.chunk_overlap_chars)
        if len(chunks) > self._settings.max_chunks:
            raise DocumentError(
                f"document needs {len(chunks)} chunks; the limit is {self._settings.max_chunks} (MAX_CHUNKS)"
            )
        return chunks

    async def _describe(self, page: Page, visual: Visual, title: str) -> ImageAnalysis | None:
        try:
            return await self._llm.analyze_image(
                visual.data,
                visual.mime_type,
                context=f'page {page.number} of the document "{title}". '
                f"Text on that page starts with: {page.text[:VISION_CONTEXT_TEXT_CHARS]!r}",
            )
        except (LLMNotConfiguredError, LLMUnavailableError):
            raise
        except LLMError as exc:
            logger.warning("Visual on page %d of %r skipped: %s", page.number, title, exc)
            return None

    async def _build_text(self, document: ParsedDocument, title: str) -> str:
        visuals = [(page, visual) for page in document.pages for visual in page.visuals]
        analyses = await _run_all([self._describe(page, visual, title) for page, visual in visuals])
        skipped = sum(a is None for a in analyses)
        if skipped:
            logger.warning("%d of %d visuals skipped for %r", skipped, len(visuals), title)
        if visuals and all(a is None for a in analyses) and not any(page.text for page in document.pages):
            raise LLMError("every visual description failed")
        descriptions: dict[int, list[str]] = {}
        for (page, _), analysis in zip(visuals, analyses, strict=True):
            if analysis and analysis.is_relevant and analysis.description.strip():
                descriptions.setdefault(page.number, []).append(analysis.description.strip())

        sections = []
        for page in document.pages:
            parts = [page.text] if page.text else []
            parts.extend(f"[Visual content: {d}]" for d in descriptions.get(page.number, []))
            if parts:
                sections.append("\n\n".join(parts))
        return "\n\n".join(sections)

    async def _extract(self, text: str) -> ExtractionResult:
        chunks = self._chunks(text)
        used = [name for name, _ in await self._repository.list_relation_types()]
        known_types = [*CANONICAL_RELATION_TYPES, *sorted(set(used) - set(CANONICAL_RELATION_TYPES))]
        known_types = known_types[:KNOWN_RELATION_TYPES_IN_PROMPT]
        results = await _run_all([self._llm.extract(chunk, known_types) for chunk in chunks])
        logger.info("Extracted from %d chunks", len(chunks))
        return merge_extractions(results)

    async def _persist(
        self, *, article_id: str, title: str, source_type: SourceType, text: str, extraction: ExtractionResult
    ) -> IngestionResult:
        type_plan, character_plan = await _run_all(
            [
                self._relation_types.plan([r.type for r in extraction.relations]),
                self._characters.plan(extraction.characters),
            ]
        )

        async def work(writer: GraphWriter) -> tuple[list[CharacterRecord], list[RelationRecord]]:
            await writer.complete_article(article_id, text)
            resolved = await self._characters.apply(character_plan, writer)
            characters = list({record.id: record for record in resolved.values()}.values())
            await writer.link_characters_to_article(article_id, [c.id for c in characters])
            relations = _build_relations(extraction, resolved, type_plan)
            await self._relation_types.create_used(type_plan, {r.type for r in relations}, writer)
            await writer.upsert_relations(
                article_id,
                [
                    {
                        "source_id": r.source_id,
                        "target_id": r.target_id,
                        "type": r.type,
                        "description": r.description,
                    }
                    for r in relations
                ],
            )
            return characters, relations

        characters, relations = await self._repository.write_transaction(work)
        return IngestionResult(
            article_id=article_id,
            title=title,
            source_type=source_type.value,
            duplicate=False,
            characters=sorted(characters, key=lambda c: c.name),
            relations=relations,
        )


def _build_relations(
    extraction: ExtractionResult, resolved: dict[str, CharacterRecord], type_plan: TypePlan
) -> list[RelationRecord]:
    relations: dict[tuple[frozenset[str] | tuple[str, str], str], RelationRecord] = {}
    for relation in extraction.relations:
        source, target = resolved[relation.source], resolved[relation.target]
        # Two extracted names may resolve to one stored character.
        if source.id == target.id:
            continue
        relation_type = type_plan.mapping[relation.type]
        # A symmetric type has no direction: keep the first-seen orientation and drop the mirrored duplicate.
        ends = (
            frozenset({source.id, target.id}) if relation_type in SYMMETRIC_RELATION_TYPES else (source.id, target.id)
        )
        relations.setdefault(
            (ends, relation_type),
            RelationRecord(
                source_id=source.id,
                source_name=source.name,
                target_id=target.id,
                target_name=target.name,
                type=relation_type,
                description=relation.description,
            ),
        )
    return list(relations.values())

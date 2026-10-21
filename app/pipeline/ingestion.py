import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePath

from app.core.config import Settings
from app.graph.repository import (
    ArticleRecord,
    ArticleStatus,
    CharacterRecord,
    GraphRepository,
    RelationRecord,
)
from app.llm.base import LLMClient, LLMError
from app.llm.schemas import ExtractionResult, ImageAnalysis
from app.llm.vocabulary import CANONICAL_RELATION_TYPES, SYMMETRIC_RELATION_TYPES
from app.pipeline.chunking import chunk_text
from app.pipeline.documents import DocumentError, ParsedDocument, PdfOptions, parse_markdown, parse_pdf
from app.pipeline.merging import merge_extractions
from app.pipeline.resolution import CharacterResolver, RelationTypeResolver, TypePlan

logger = logging.getLogger(__name__)

KNOWN_RELATION_TYPES_IN_PROMPT = 100
# A "processing" article older than this is a leftover of a crashed worker and may be reprocessed.
STALE_PROCESSING_SECONDS = 15 * 60
VISION_CONTEXT_TEXT_CHARS = 500


class SourceType(StrEnum):
    MARKDOWN = "markdown"
    PDF = "pdf"


class ArticleInProgressError(Exception):
    pass


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


def unwrap_plan_failure(failure: ExceptionGroup) -> BaseException:
    """Logs every failed plan and picks the one to raise: an LLMError (502/503) if any, else the first."""
    for exc in failure.exceptions:
        logger.error("Resolution planning failed: %r", exc, exc_info=exc)
    return next((exc for exc in failure.exceptions if isinstance(exc, LLMError)), failure.exceptions[0])


class ArticleIngestionService:
    """parse -> describe visuals -> chunk -> extract -> merge -> plan resolution (reads + LLM judges) -> persist.
    All LLM calls finish before the first graph write, except the idempotent vocabulary seed."""

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
        self._pdf_options = PdfOptions(
            max_visual_items=settings.pdf_max_visual_items,
            min_image_side_px=settings.pdf_min_image_side_px,
            render_dpi=settings.pdf_render_dpi,
            scanned_page_max_chars=settings.pdf_scanned_page_max_chars,
            drawings_render_threshold=settings.pdf_drawings_render_threshold,
        )

    async def ingest(self, upload: UploadedDocument) -> IngestionResult:
        content_hash = hashlib.sha256(upload.content).hexdigest()
        existing = await self._repository.find_article_by_hash(content_hash)
        if existing is not None:
            if existing.status is ArticleStatus.COMPLETED:
                return await self._existing_result(existing)
            if time.time() - existing.created_at_epoch < STALE_PROCESSING_SECONDS:
                raise ArticleInProgressError(existing.id)

        document = await self._parse(upload)
        title = upload.title or document.title or (PurePath(upload.filename).stem if upload.filename else None)
        text = await self._build_text(document, title or "untitled")
        if not text.strip():
            raise DocumentError("no text or relevant visual content could be extracted")

        extraction = await self._extract(text)
        if existing is not None:
            logger.warning("Reprocessing stale article %s", existing.id)
            await self._repository.delete_article(existing.id)
        return await self._persist(
            title=title or "Untitled",
            source_type=upload.source_type,
            content_hash=content_hash,
            text=text,
            extraction=extraction,
        )

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

    async def _build_text(self, document: ParsedDocument, title: str) -> str:
        visuals = [(page, visual) for page in document.pages for visual in page.visuals]
        analyses: list[ImageAnalysis] = await asyncio.gather(
            *(
                self._llm.analyze_image(
                    visual.data,
                    visual.mime_type,
                    context=f'page {page.number} of the document "{title}". '
                    f"Text on that page starts with: {page.text[:VISION_CONTEXT_TEXT_CHARS]!r}",
                )
                for page, visual in visuals
            )
        )
        descriptions: dict[int, list[str]] = {}
        for (page, _), analysis in zip(visuals, analyses, strict=True):
            if analysis.is_relevant and analysis.description.strip():
                descriptions.setdefault(page.number, []).append(analysis.description.strip())

        sections = []
        for page in document.pages:
            parts = [page.text] if page.text else []
            parts.extend(f"[Visual content: {d}]" for d in descriptions.get(page.number, []))
            if parts:
                sections.append("\n\n".join(parts))
        return "\n\n".join(sections)

    async def _extract(self, text: str) -> ExtractionResult:
        chunks = chunk_text(text, self._settings.chunk_size_chars, self._settings.chunk_overlap_chars)
        used = [name for name, _ in await self._repository.list_relation_types()]
        known_types = [*CANONICAL_RELATION_TYPES, *sorted(set(used) - set(CANONICAL_RELATION_TYPES))]
        known_types = known_types[:KNOWN_RELATION_TYPES_IN_PROMPT]
        results = await asyncio.gather(*(self._llm.extract(chunk, known_types) for chunk in chunks))
        logger.info("Extracted from %d chunks", len(chunks))
        return merge_extractions(list(results))

    async def _persist(
        self, *, title: str, source_type: SourceType, content_hash: str, text: str, extraction: ExtractionResult
    ) -> IngestionResult:
        try:
            async with asyncio.TaskGroup() as group:  # cancels the sibling plan if one fails
                type_task = group.create_task(self._relation_types.plan([r.type for r in extraction.relations]))
                character_task = group.create_task(self._characters.plan(extraction.characters))
        except ExceptionGroup as failure:
            # Plain `except`, not `except*`: raising inside `except*` would wrap the error in a group again.
            raise unwrap_plan_failure(failure) from None
        type_plan, character_plan = type_task.result(), character_task.result()

        article_id = str(uuid.uuid4())
        await self._repository.create_article(
            article_id=article_id, title=title, source_type=source_type.value, content_hash=content_hash, text=text
        )
        try:
            resolved = await self._characters.apply(character_plan)
            characters = list({record.id: record for record in resolved.values()}.values())
            await self._repository.link_characters_to_article(article_id, [c.id for c in characters])
            relations = _build_relations(extraction, resolved, type_plan)
            await self._relation_types.create_used(type_plan, {r.type for r in relations})
            await self._repository.upsert_relations(
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
            await self._repository.mark_article_completed(article_id)
        except Exception:
            logger.exception("Persisting article %s failed, rolling it back", article_id)
            await self._repository.delete_article(article_id)
            raise

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

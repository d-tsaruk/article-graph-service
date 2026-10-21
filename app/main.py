import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from neo4j import AsyncGraphDatabase, NotificationDisabledClassification

from app.api.deps import Services
from app.api.routes import articles, auth, characters, health, relations
from app.core.config import Settings
from app.core.security import TokenService
from app.graph.repository import DuplicateArticleError, GraphRepository
from app.llm.base import LLMClient, LLMError, LLMNotConfiguredError, UnconfiguredLLMClient
from app.llm.openai_client import build_openai_client
from app.pipeline.documents import DocumentError
from app.pipeline.ingestion import ArticleIngestionService, ArticleInProgressError
from app.pipeline.resolution import RelationTypeResolver

logger = logging.getLogger(__name__)


def _default_llm_client(settings: Settings) -> LLMClient:
    if settings.openai_api_key is None:
        logger.warning("OPENAI_API_KEY is not set: uploads are disabled, read endpoints keep working")
        return UnconfiguredLLMClient()
    return build_openai_client(settings)


def _error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DocumentError)
    async def document_error(_: Request, exc: DocumentError) -> JSONResponse:
        return _error(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc))

    @app.exception_handler(LLMNotConfiguredError)
    async def llm_not_configured(_: Request, exc: LLMNotConfiguredError) -> JSONResponse:
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))

    @app.exception_handler(LLMError)
    async def llm_error(_: Request, exc: LLMError) -> JSONResponse:
        logger.error("LLM failure: %s", exc)
        return _error(status.HTTP_502_BAD_GATEWAY, f"LLM backend error: {exc}")

    @app.exception_handler(ArticleInProgressError)
    @app.exception_handler(DuplicateArticleError)
    async def in_progress(_: Request, exc: Exception) -> JSONResponse:
        return _error(status.HTTP_409_CONFLICT, "An article with identical content is being processed")


def create_app(settings: Settings | None = None, llm_client: LLMClient | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
            # On an empty database every query warns that properties like `aliases` do not exist yet.
            notifications_disabled_classifications=[NotificationDisabledClassification.UNRECOGNIZED],
        )
        try:
            await driver.verify_connectivity()
            repository = GraphRepository(driver, settings.neo4j_database, settings.embedding_dimensions)
            await repository.init_schema()
            llm = llm_client or _default_llm_client(settings)
            relation_types = RelationTypeResolver(repository, llm, match_score=settings.relation_type_match_score)
            app.state.services = Services(
                settings=settings,
                tokens=TokenService(settings),
                repository=repository,
                ingestion=ArticleIngestionService(repository, llm, settings, relation_types),
                relation_types=relation_types,
            )
            yield
        finally:
            await driver.close()

    app = FastAPI(
        title="Article Graph Service",
        version="0.1.0",
        description="Extracts characters and their relations from articles into a Neo4j knowledge graph.",
        lifespan=lifespan,
    )
    _register_error_handlers(app)
    for module in (health, auth, articles, characters, relations):
        app.include_router(module.router)
    return app
